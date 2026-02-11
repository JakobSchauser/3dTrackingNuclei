import zarr
import numpy as np
import cupy as cp
from ultrack import MainConfig, Tracker
from ultrack.utils.array import create_zarr, array_apply
from ultrack.imgproc import detect_foreground, robust_invert
from ultrack.imgproc.flow import timelapse_flow
from ultrack.utils import labels_to_contours
from cellpose.models import CellposeModel
import pathlib
import os
import tifffile


class CellTrackingPipeline:
    def __init__(self, data_path: str, savepath : str, config : MainConfig, use_gpu : bool):
        self.config = config
        self.use_gpu = use_gpu

        self.data = None 
        self.foreground = None
        self.contours = None
        self.flow = None

        self.data_path = pathlib.Path(data_path)
        self.savepath = pathlib.Path(savepath)
        self.savepath.mkdir(parents=True, exist_ok=True)


    def load_data(self, subsample : int) -> zarr.Array:

        print("Loading images...")
        imgs = []
        for path in os.listdir(self.data_path):
            if not path.endswith(".tif"):
                print(f"Skipping {path}, not a .tif file.")
                continue

            print(f"Loading {path}...")
            image = tifffile.imread(self.data_path / path)
            imgs.append(image[::subsample,::subsample, ::subsample])

        print("")

        self.data = np.array(imgs)  # shape [T, Z, Y, X]
        print(f"Loaded data shape: {self.data.shape}")

    def segment(self, cellpose : bool) -> None:
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.use_gpu:
            # check if gpu is available
            try:
                _ = cp.zeros(1)  # try to allocate a small array on GPU
                print("GPU is available and will be used for segmentation.")
                print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')}")
            except cp.cuda.runtime.CUDARuntimeError:
                raise RuntimeError("GPU is not available. Please check your CUDA installation or set use_gpu=False.")

        if cellpose:
            # Use pretrained 3D Cellpose for nuclei segmentation on GPU
            model = CellposeModel(gpu=self.use_gpu)  # pretrained model

            masks = np.zeros_like(self.data, dtype=np.uint16)  # to store segmentation results

            for t in range(self.data.shape[0]):
                print(f"Segmenting frame {t+1}/{self.data.shape[0]} with Cellpose...")
                img_3D = self.data[t]  # shape [Z, Y, X]
                msk, _, _ = model.eval(img_3D, z_axis=1, channel_axis=1, batch_size=32, do_3D=True, flow3D_smooth=1)
                masks[t] = msk.astype(np.uint16)  # store segmentation mask for this frame

            # Convert labels to Ultrack input: foreground and contour confidence map
            foreground, contours = labels_to_contours(masks)
            # save intermediate results to disk
            create_zarr(foreground.shape, np.uint8, store_or_path= self.savepath / "foreground.zarr", overwrite=True)[...] = foreground
            create_zarr(contours.shape, np.float32, store_or_path= self.savepath / "contours.zarr", overwrite=True)[...] = contours
        else:
            # Fallback: simple threshold and inversion (CPU or GPU)
            foreground = create_zarr(self.data.shape, np.uint8, store_or_path= self.savepath / "foreground.zarr", overwrite=True)
            array_apply(self.data, foreground, func=detect_foreground)  # bool array of cells
            contours = create_zarr(self.data.shape, np.float32, store_or_path= self.savepath / "contours.zarr", overwrite=True)
            array_apply(self.data, contours, func=robust_invert)        # boundaries as float map

            flow = timelapse_flow(self.data)  # returns vector field [T, Z, Y, X, 3] of displacements
        create_zarr(flow.shape, np.float32, store_or_path= self.savepath / "flow.zarr", overwrite=True)[...] = flow

        # close zarr files to flush to disk
        del foreground, contours, flow,
    
    def load_segmentation(self) -> None:
        self.foreground = zarr.open(self.savepath / "foreground.zarr", mode='r')
        self.contours = zarr.open(self.savepath / "contours.zarr", mode='r')
        self.flow = zarr.open(self.savepath / "flow.zarr", mode='r')
        print("Loaded segmentation and flow from disk.")

    def track(self) -> None:
        if self.foreground is None or self.contours is None or self.flow is None:
            raise ValueError("Segmentation and flow data not loaded. Call load_segmentation() first.")

        tracker = Tracker(self.config)
        tracker.add_flow(vector_field=self.flow)  # incorporates estimated movement between frames
        
        print("Tracking cells...")
        tracker.track(foreground=self.foreground, edges=self.contours)  # edges = contour map

        print("Done!")
        print("Saving tracks to CSV...")
        tracks_df, _ = tracker.to_tracks_layer()
        tracks_df = tracks_df[["track_id", "t", "z", "y", "x"]]
        tracks_df.to_csv(self.savepath / "tracked_nuclei.csv", index=False)

        print("All done")
