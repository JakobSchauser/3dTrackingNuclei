import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
	import zarr
except ImportError:  # zarr is optional for background loading
	zarr = None


def ensure_outdir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def load_detection(path: Path):
	"""Load detection.zarr if available, otherwise return None."""
	if not path.exists():
		print(f"No detection volume at {path}, rendering on blank background.")
		return None
	if zarr is None:
		print("zarr is not installed; skipping detection background.")
		return None
	try:
		arr = zarr.open(path, mode="r")
	except Exception as exc:  # keep visualization running even if zarr fails
		print(f"Could not open {path}: {exc}")
		return None
	if getattr(arr, "ndim", 0) != 4:
		print(f"Expected 4D array (t, z, y, x); got shape {getattr(arr, 'shape', None)}")
		return None
	return arr


def compute_bounds(df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
	x_min, x_max = df["x"].min(), df["x"].max()
	y_min, y_max = df["y"].min(), df["y"].max()
	span = max(x_max - x_min, y_max - y_min, 1.0)
	pad = 0.05 * span
	return (x_min - pad, x_max + pad), (y_min - pad, y_max + pad)


def compute_bounds_3d(df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
	x_min, x_max = df["x"].min(), df["x"].max()
	y_min, y_max = df["y"].min(), df["y"].max()
	z_min, z_max = df["z"].min(), df["z"].max()
	span = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
	pad = 0.05 * span
	return (
		(x_min - pad, x_max + pad),
		(y_min - pad, y_max + pad),
		(z_min - pad, z_max + pad),
	)


def build_color_map(track_ids: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
	cmap = plt.cm.get_cmap("tab20", len(track_ids))
	return {tid: cmap(i) for i, tid in enumerate(sorted(track_ids))}


def save_video(frames, video_path: Path, fps: int) -> None:
	"""Save frames to MP4 via FFMPEG; raise helpful error if missing."""
	try:
		with imageio.get_writer(video_path, format="FFMPEG", fps=fps, codec="libx264") as writer:
			for frame in frames:
				writer.append_data(frame)
	except Exception as exc:
		raise RuntimeError(
			"Could not write MP4. Install ffmpeg or `pip install imageio[ffmpeg]` and retry."
		) from exc


def render_frames(
	df: pd.DataFrame,
	detection,
	outdir: Path,
	fps: int,
	dpi: int,
	marker_size: float,
	alpha: float,
	annotate: bool,
):
	if df.empty:
		raise ValueError("tracks.csv is empty")

	ensure_outdir(outdir)
	df = df.sort_values(["t", "track_id"]).reset_index(drop=True)
	xlim, ylim = compute_bounds(df)
	colors = build_color_map(df["track_id"].unique())

	frames = []
	for t, frame_df in df.groupby("t"):
		fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

		background = None
		if detection is not None:
			try:
				background = np.asarray(detection[int(t)]).max(axis=0)
			except Exception as exc:
				print(f"Skipping background for t={t}: {exc}")

		if background is not None:
			ax.imshow(background, cmap="gray", origin="lower")
		else:
			ax.set_facecolor("black")

		xy_colors = [colors[tid] for tid in frame_df["track_id"]]
		ax.scatter(
			frame_df["x"],
			frame_df["y"],
			c=xy_colors,
			s=marker_size,
			alpha=alpha,
			edgecolor="white",
			linewidth=0.5,
		)

		if annotate:
			for _, row in frame_df.iterrows():
				ax.text(
					row["x"],
					row["y"],
					str(int(row["track_id"])),
					color="white",
					fontsize=6,
					ha="center",
					va="center",
				)

		ax.set_title(f"t = {int(t)} | n = {len(frame_df)}")
		ax.set_xlim(*xlim)
		ax.set_ylim(*ylim)
		ax.set_aspect("equal")
		ax.axis("off")

		fig.tight_layout()
		frame_path = outdir / f"frame_{int(t):04d}.png"
		fig.savefig(frame_path)
		plt.close(fig)

		frames.append(imageio.imread(frame_path))

	if not frames:
		raise ValueError("No frames were rendered; check tracks.csv content")

	video_path = outdir / "tracks.mp4"
	save_video(frames, video_path, fps)
	print(f"Saved {len(frames)} frames to {outdir} and video to {video_path}")


def render_frames_3d(
	df: pd.DataFrame,
	outdir: Path,
	fps: int,
	dpi: int,
	marker_size: float,
	alpha: float,
	annotate: bool,
	elev: float,
	azim_base: float,
	azim_speed: float,
):
	if df.empty:
		raise ValueError("tracks.csv is empty")

	ensure_outdir(outdir)
	df = df.sort_values(["t", "track_id"]).reset_index(drop=True)
	xlim, ylim, zlim = compute_bounds_3d(df)
	colors = build_color_map(df["track_id"].unique())

	frames = []
	for idx, (t, frame_df) in enumerate(df.groupby("t")):
		fig = plt.figure(figsize=(6, 6), dpi=dpi)
		ax = fig.add_subplot(111, projection="3d")

		xy_colors = [colors[tid] for tid in frame_df["track_id"]]
		ax.scatter(
			frame_df["x"],
			frame_df["y"],
			frame_df["z"],
			c=xy_colors,
			s=marker_size,
			alpha=alpha,
			edgecolor="white",
			linewidth=0.5,
		)

		if annotate:
			for _, row in frame_df.iterrows():
				ax.text(
					row["x"],
					row["y"],
					row["z"],
					str(int(row["track_id"])),
					color="white",
					fontsize=6,
					ha="center",
					va="center",
				)

		ax.set_xlim(*xlim)
		ax.set_ylim(*ylim)
		ax.set_zlim(*zlim)
		ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
		azim_angle = azim_base if azim_speed == 0 else azim_base + azim_speed * idx
		ax.view_init(elev=elev, azim=azim_angle)
		ax.set_title(f"t = {int(t)} | n = {len(frame_df)}")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		ax.grid(False)
		ax.set_facecolor("black")

		fig.tight_layout()
		frame_path = outdir / f"frame3d_{int(t):04d}.png"
		fig.savefig(frame_path)
		plt.close(fig)

		frames.append(imageio.imread(frame_path))

	if not frames:
		raise ValueError("No 3D frames were rendered; check tracks.csv content")

	video_path = outdir / "tracks3d.mp4"
	save_video(frames, video_path, fps)
	print(f"Saved {len(frames)} 3D frames to {outdir} and video to {video_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Render 2D/3D track visualizations.")
	parser.add_argument("--tracks", type=Path, default=Path("tracks.csv"), help="Path to tracks.csv")
	parser.add_argument("--outdir", type=Path, default=Path("ultrack_plots"), help="Output directory for frames and videos")
	parser.add_argument("--detection", type=Path, default=Path("detection.zarr"), help="Optional detection.zarr volume for background")
	parser.add_argument("--fps", type=int, default=4, help="Frames per second for videos")
	parser.add_argument("--dpi", type=int, default=150, help="Matplotlib figure DPI")
	parser.add_argument("--annotate", action="store_true", help="Annotate points with track ids")
	parser.add_argument("--marker2d", type=float, default=30.0, help="Marker size for 2D scatter")
	parser.add_argument("--marker3d", type=float, default=30.0, help="Marker size for 3D scatter")
	parser.add_argument("--alpha", type=float, default=0.9, help="Marker alpha")
	parser.add_argument("--elev", type=float, default=20.0, help="Elevation angle for 3D view")
	parser.add_argument("--azim", type=float, default=45.0, help="Base azimuth angle for 3D view")
	parser.add_argument("--azim-speed", type=float, default=0.0, help="Azimuth step per frame for 3D rotation (0 keeps camera steady)")
	return parser.parse_args()


def main():
	args = parse_args()
	print(f"Loading tracks from {args.tracks}...")
	if not args.tracks.exists():
		raise FileNotFoundError(f"tracks file not found: {args.tracks}")

	df = pd.read_csv(args.tracks)
	required_cols = {"track_id", "t", "z", "y", "x"}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"tracks.csv missing columns: {sorted(missing)}")

	# flip z
	df["z"] = -df["z"]

	# remove all odd-numbered track IDs for clearer visualization
	# df = df[df["track_id"] % 2 == 0].reset_index(drop=True)

	detection = load_detection(args.detection)
	render_frames(df, detection, args.outdir, args.fps, args.dpi, args.marker2d, args.alpha, args.annotate)
	render_frames_3d(
		df,
		args.outdir,
		args.fps,
		args.dpi,
		args.marker3d,
		args.alpha,
		args.annotate,
		args.elev,
		args.azim,
		args.azim_speed,
	)

	print("Visualization complete.")


if __name__ == "__main__":
	main()
