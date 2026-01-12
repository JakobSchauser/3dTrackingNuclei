import tifffile
import numpy as np
from ultrack import MainConfig, load_config, Tracker, track, to_tracks_layer, tracks_to_zarr
from ultrack.imgproc import robust_invert, detect_foreground
from ultrack.utils.array import array_apply, create_zarr
from ultrack.utils.cuda import to_cpu, on_gpu

# load in image
p = "data/MATTEO TO ALA/Embryo_37_intrareg_fuse_t0"
paths = [str(i) for i in range(75,98)]  # list of timepoints


print("Loading images...")
imgs = []
for path in paths:
  print(f"Loading {path}...")
  image = tifffile.imread(p + path + ".tif")
  imgs.append(image[::4,::4, ::4])


images = np.array(imgs)


print("")
detection = create_zarr(images.shape, bool, store_or_path="detection.zarr", overwrite=True)


voxel_size = [1,1,1]


print("Detecting foreground...")

array_apply(
    images,
    out_array=detection,
    func=on_gpu(detect_foreground),
    sigma=25.0,
    voxel_size=voxel_size,
)
print("")

print("Computing boundaries...")

boundaries = create_zarr(images.shape, np.float16, store_or_path="boundaries.zarr", overwrite=True)
array_apply(
    images,
    out_array=boundaries,
    func=on_gpu(robust_invert),
    voxel_size=voxel_size,
)


cfg =  MainConfig()  # or load default config
cfg.segmentation_config.threshold = 0.5
print("")

print("Tracking...")
track(
    cfg,
    foreground=detection,
    edges=boundaries,
    scale=voxel_size,
    overwrite=True,
)

print("Exporting tracks...")

# export to good format 
tracks_df, graph = to_tracks_layer(cfg)

tracks_df = to_cpu(tracks_df)

tracks_df.to_csv("tracks.csv", index=False)

