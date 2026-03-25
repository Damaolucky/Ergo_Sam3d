#!/usr/bin/env python3
"""Group sample artifacts into one clip folder and build scene geometry files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline_utils import (
    DEFAULT_OUTPUTS_DIR,
    backproject_depth_to_pointcloud,
    ensure_output_roots,
    load_json,
    load_pickle,
    pointcloud_colors_from_rgb,
    resize_rgb_to_shape,
    resolve_in_outputs,
    save_pointcloud_preview,
    save_pointcloud_ply,
    valid_depth_mask,
    write_json,
)


def resolve_manifest_path(manifest_arg: str) -> Path:
    """Resolve a manifest JSON path directly or from the outputs root."""
    return resolve_in_outputs(manifest_arg, label="Manifest")


def summarize_depth(depth_meters: np.ndarray) -> dict:
    """Summarize depth coverage and basic metric statistics."""
    arr = np.asarray(depth_meters, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    stats = {
        "shape": list(arr.shape),
        "num_pixels": int(arr.size),
        "num_valid_pixels": int(valid.sum()),
        "valid_ratio": float(valid.mean()),
    }

    if valid.any():
        values = arr[valid]
        stats.update(
            {
                "depth_min_m": float(values.min()),
                "depth_max_m": float(values.max()),
                "depth_mean_m": float(values.mean()),
                "depth_median_m": float(np.median(values)),
            }
        )
    else:
        stats.update(
            {
                "depth_min_m": None,
                "depth_max_m": None,
                "depth_mean_m": None,
                "depth_median_m": None,
            }
        )
    return stats


def summarize_pointcloud(points: np.ndarray) -> dict:
    """Summarize scene point count and axis-aligned bounds."""
    if points.shape[0] == 0:
        return {"num_points": 0, "bbox_min": None, "bbox_max": None}

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return {
        "num_points": int(points.shape[0]),
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
    }


def move_into_clip_dir(src: Path, dst_dir: Path) -> Path:
    """Move one artifact into the clip folder, tolerating repeated runs."""
    dst = dst_dir / src.name

    if src.resolve() == dst.resolve():
        return dst

    if dst.exists():
        try:
            if src.exists():
                src.unlink()
        except Exception:
            pass
        return dst

    shutil.move(str(src), str(dst))
    return dst


def copy_into_clip_dir(src: Path, dst_dir: Path) -> Path:
    """Copy one shared artifact into the clip folder without deleting the source."""
    dst = dst_dir / src.name

    if src.resolve() == dst.resolve():
        return dst

    if dst.exists():
        return dst

    shutil.copy2(str(src), str(dst))
    return dst


def main() -> None:
    """Parse arguments and build a clip-centered geometry workspace."""
    parser = argparse.ArgumentParser(
        description="Organize one sample into its own output folder and prepare geometry artifacts."
    )
    parser.add_argument(
        "manifest",
        help=(
            "Sample manifest filename or path, e.g. "
            "2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4.sample_manifest.json"
        ),
    )
    args = parser.parse_args()

    ensure_output_roots()

    manifest_path = resolve_manifest_path(args.manifest)
    manifest = load_json(manifest_path)

    clip_name = manifest["clip_name"]
    clip_dir = DEFAULT_OUTPUTS_DIR / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    mapping_json = Path(manifest["mapping_json"])
    rgb_png = Path(manifest["outputs"]["rgb_png"])
    depth_raw_npy = Path(manifest["outputs"]["depth_raw_npy"])
    depth_meters_npy = Path(manifest["outputs"]["depth_meters_npy"])
    depth_vis_png = Path(manifest["outputs"]["depth_vis_png"])
    intrinsics_pkl = Path(manifest["outputs"]["intrinsics_pkl"])

    copied_mapping = copy_into_clip_dir(mapping_json, clip_dir)
    copied_manifest = move_into_clip_dir(manifest_path, clip_dir)
    copied_rgb = move_into_clip_dir(rgb_png, clip_dir)
    copied_depth_raw = move_into_clip_dir(depth_raw_npy, clip_dir)
    copied_depth_m = move_into_clip_dir(depth_meters_npy, clip_dir)
    copied_depth_vis = move_into_clip_dir(depth_vis_png, clip_dir)
    copied_intr = move_into_clip_dir(intrinsics_pkl, clip_dir)

    rgb = np.asarray(Image.open(copied_rgb).convert("RGB"))
    depth_meters = np.load(copied_depth_m)
    intrinsics = load_pickle(copied_intr)
    valid_mask = valid_depth_mask(depth_meters)
    depth_stats = summarize_depth(depth_meters)
    rgb = resize_rgb_to_shape(rgb, depth_meters.shape)
    pointcloud, pointcloud_uv = backproject_depth_to_pointcloud(
        depth_meters,
        intrinsics,
        valid_mask=valid_mask,
        return_uv=True,
    )
    pointcloud_colors = pointcloud_colors_from_rgb(rgb, pointcloud_uv)
    pointcloud_stats = summarize_pointcloud(pointcloud)

    pointcloud_npy = clip_dir / "pointcloud.npy"
    pointcloud_ply = clip_dir / "pointcloud_rgb.ply"
    pointcloud_preview_png = clip_dir / "pointcloud_preview.png"
    geometry_stats_json = clip_dir / "geometry_stats.json"

    np.save(pointcloud_npy, pointcloud)
    save_pointcloud_ply(pointcloud_ply, pointcloud, colors=pointcloud_colors)
    save_pointcloud_preview(
        pointcloud,
        pointcloud_preview_png,
        title="Point Cloud Preview",
        empty_message="No valid 3D points",
        colors=pointcloud_colors,
    )

    geometry_stats = {
        "clip_name": clip_name,
        "source_clip_name": manifest.get("source_clip_name", clip_name),
        "sample_name": manifest.get("sample_name", clip_name),
        "sample_role": manifest.get("sample_role"),
        "position_label": manifest.get("position_label"),
        "clip_time_seconds": manifest.get("clip_time_seconds"),
        "camera": manifest["camera"],
        "source_video": manifest["source_video"],
        "source_start": manifest["source_start"],
        "source_end": manifest["source_end"],
        "source_duration": manifest["source_duration"],
        "depth_stats": depth_stats,
        "depth_filter": {
            "valid_points_before_filter": int((np.isfinite(depth_meters) & (depth_meters > 0)).sum()),
            "pointcloud_points_after_filter": int(pointcloud.shape[0]),
            "max_depth_percentile": 99.8,
            "max_depth_m": float(depth_meters[valid_mask].max()) if pointcloud.shape[0] else None,
        },
        "pointcloud_stats": pointcloud_stats,
        "files": {
            "mapping_json": str(copied_mapping),
            "sample_manifest_json": str(copied_manifest),
            "rgb_png": str(copied_rgb),
            "depth_raw_npy": str(copied_depth_raw),
            "depth_meters_npy": str(copied_depth_m),
            "depth_vis_png": str(copied_depth_vis),
            "intrinsics_pkl": str(copied_intr),
            "pointcloud_npy": str(pointcloud_npy),
            "pointcloud_rgb_ply": str(pointcloud_ply),
            "pointcloud_preview_png": str(pointcloud_preview_png),
        },
    }
    write_json(geometry_stats_json, geometry_stats)
    print(json.dumps(geometry_stats, indent=2))


if __name__ == "__main__":
    main()
