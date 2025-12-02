#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import tifffile

# Ensure local imports work regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from import_data import load_tiff_files

try:
    from cellpose import models
except ImportError as e:
    raise ImportError("cellpose not installed. Install with: pip install 'cellpose[all]'") from e


def infer_axes(volume: np.ndarray, default_z: int | None, default_channel_axis: int | None):
    if default_z is not None or default_channel_axis is not None:
        return default_z, default_channel_axis
    if volume.ndim == 3:
        return 0, None
    if volume.ndim == 4:
        return 0, -1
    raise ValueError(f"Unsupported volume ndim {volume.ndim}; expected 3D or 4D.")


def run_segmentation(data_dir: str,
                     out_masks_dir: str,
                     out_checkpoints_dir: str,
                     model_type: str,
                     channels: list[int],
                     gpu: bool,
                     diameter: float | None,
                     z_axis: int | None,
                     channel_axis: int | None,
                     save_centroids: bool = True,
                     do_linking: bool = True):
    os.makedirs(out_masks_dir, exist_ok=True)
    os.makedirs(out_checkpoints_dir, exist_ok=True)

    images = load_tiff_files(data_dir)
    print(f"Loaded {len(images)} volumes from {data_dir}.")
    if not images:
        print("No .tif files found. Exiting without work.")
        return 0

    model = models.CellposeModel(gpu=gpu, model_type=model_type)

    masks_dict: dict[str, np.ndarray] = {}

    for i, (name, vol) in enumerate(sorted(images.items())):
        print(f"Segmenting volume {i+1}/{len(images)}: {name} shape={vol.shape}")
        v = vol.astype(np.float32)
        z_ax, ch_ax = infer_axes(v, z_axis, channel_axis)

        masks, flows, styles = model.eval(
            v,
            channels=channels,
            do_3D=True,
            diameter=diameter,
            channel_axis=ch_ax,
            z_axis=z_ax,
        )
        masks_dict[name] = masks

        base = os.path.splitext(name)[0]
        mask_path = os.path.join(out_masks_dir, f"{base}_cellpose_mask.tif")
        tifffile.imwrite(mask_path, masks.astype(np.uint16))
        print(f"Saved mask: {mask_path} (labels: {int(masks.max())})")

    if save_centroids:
        centroids = []
        for frame_idx, (name, masks) in enumerate(sorted(masks_dict.items())):
            labels = np.unique(masks)
            labels = labels[labels != 0]
            for lab in labels:
                coords = np.argwhere(masks == lab)
                z_mean, y_mean, x_mean = coords.mean(axis=0)
                centroids.append({
                    'frame': frame_idx,
                    'volume': os.path.splitext(name)[0],
                    'particle': int(lab),
                    'z': float(z_mean),
                    'y': float(y_mean),
                    'x': float(x_mean),
                })
        centroids_df = pd.DataFrame(centroids)
        centroids_out = os.path.join(out_checkpoints_dir, 'cellpose_centroids.csv')
        centroids_df.to_csv(centroids_out, index=False)
        print(f"Saved centroids to {centroids_out} (N={len(centroids_df)})")

        if do_linking and len(centroids_df['frame'].unique()) >= 2:
            frames_available = sorted(centroids_df['frame'].unique())
            f0 = centroids_df[centroids_df.frame == frames_available[0]].copy()
            f1 = centroids_df[centroids_df.frame == frames_available[1]].copy()
            coords0 = f0[['z','y','x']].values
            coords1 = f1[['z','y','x']].values
            linked = []
            for i, c0 in enumerate(coords0):
                dists = np.linalg.norm(coords1 - c0, axis=1)
                j = int(dists.argmin())
                linked.append({
                    'particle_f0': int(f0.iloc[i]['particle']),
                    'particle_f1': int(f1.iloc[j]['particle']),
                    'dist': float(dists[j]),
                })
            links_df = pd.DataFrame(linked)
            links_out = os.path.join(out_checkpoints_dir, 'cellpose_links.csv')
            links_df.to_csv(links_out, index=False)
            print(f"Saved links to {links_out} (N={len(links_df)})")
        else:
            print("Skipping linking or insufficient frames.")

    return len(images)


# Default configuration for simple, argument-free execution
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data'))
DEFAULT_OUT_MASKS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'docs', 'plots', 'cellpose_masks'))
DEFAULT_OUT_CHECKPOINTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'checkpoint_data'))
DEFAULT_MODEL_TYPE = 'nuclei'
DEFAULT_CHANNELS = [0, 0]
DEFAULT_GPU = True
DEFAULT_DIAMETER = None
DEFAULT_Z_AXIS = None
DEFAULT_CHANNEL_AXIS = None
DEFAULT_SAVE_CENTROIDS = True
DEFAULT_DO_LINKING = True


def main():
    n = run_segmentation(
        data_dir=DEFAULT_DATA_DIR,
        out_masks_dir=DEFAULT_OUT_MASKS_DIR,
        out_checkpoints_dir=DEFAULT_OUT_CHECKPOINTS_DIR,
        model_type=DEFAULT_MODEL_TYPE,
        channels=DEFAULT_CHANNELS,
        gpu=DEFAULT_GPU,
        diameter=DEFAULT_DIAMETER,
        z_axis=DEFAULT_Z_AXIS,
        channel_axis=DEFAULT_CHANNEL_AXIS,
        save_centroids=DEFAULT_SAVE_CENTROIDS,
        do_linking=DEFAULT_DO_LINKING,
    )
    print(f"Done. Processed {n} volume(s).")


if __name__ == '__main__':
    main()
