#!/usr/bin/env python3
"""Shared helpers for the human motion geometry preprocessing pipeline."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_WORK_ROOT = Path(
    os.environ.get("ERGO_WORK_ROOT", str(Path.home() / "hzhou"))
).expanduser()
DEFAULT_OUTPUTS_DIR = DEFAULT_WORK_ROOT / "outputs"
DEFAULT_CACHE_ROOT = DEFAULT_WORK_ROOT / "cache"
DEFAULT_DATA_ROOT = Path(
    os.environ.get("ERGO_DATA_ROOT", "/mnt/lift_data/Data")
).expanduser()
DEFAULT_CLIPS_DIR = Path(
    os.environ.get("ERGO_CLIPS_DIR", "/mnt/lift_data/Annotation/Clips")
).expanduser()
DEFAULT_JSON_ROOT = Path(
    os.environ.get("ERGO_JSON_ROOT", "/mnt/lift_data/Annotation/Annotation_json")
).expanduser()
DEFAULT_YOLO_MODEL = os.environ.get("ERGO_YOLO_MODEL", "yolov8n-seg.pt")


def ensure_output_roots() -> None:
    """Create the default local output and cache directories when needed."""
    DEFAULT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object to disk with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sanitize_label(label: str) -> str:
    """Convert a free-form label into a filesystem-safe ASCII token."""
    cleaned = [
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in str(label).strip().lower()
    ]
    token = "".join(cleaned).strip("._")
    return token or "unknown"


def format_position_label(height: Any, strength: Any, *, fallback: str) -> str:
    """Build a compact position label such as `high_24`."""
    parts: list[str] = []
    if height not in (None, "", "None"):
        parts.append(sanitize_label(str(height)))
    if strength not in (None, "", "None"):
        parts.append(sanitize_label(str(strength)))
    if not parts:
        return sanitize_label(fallback)
    return "_".join(parts)


def build_sample_output_name(clip_name: str, sample_role: str, position_label: str) -> str:
    """Build one stable sample-specific output prefix for a clip endpoint."""
    role_token = sanitize_label(sample_role)
    position_token = sanitize_label(position_label)
    return f"{clip_name}__{role_token}_{position_token}"


def load_pickle(path: Path) -> Any:
    """Load a Python pickle from disk."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, payload: Any) -> None:
    """Persist a Python object to disk as a pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def resolve_in_outputs(
    path_arg: str,
    *,
    expect_dir: bool = False,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
    label: str = "Path",
) -> Path:
    """Resolve a file or directory either directly or relative to the outputs root."""
    direct = Path(path_arg).expanduser()
    nested = outputs_dir / path_arg

    if expect_dir:
        if direct.exists() and direct.is_dir():
            return direct
        if nested.exists() and nested.is_dir():
            return nested
    else:
        if direct.exists() and direct.is_file():
            return direct
        if nested.exists() and nested.is_file():
            return nested

    raise FileNotFoundError(f"{label} not found. Tried:\n  {direct}\n  {nested}")


def resize_rgb_to_shape(rgb: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize an RGB image to match a depth map shape when needed."""
    target_h, target_w = target_shape
    if rgb.shape[:2] == target_shape:
        return rgb
    image = Image.fromarray(rgb.astype(np.uint8))
    image = image.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.asarray(image)


def robust_depth_upper_bound(
    depth_meters: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    percentile: float = 99.8,
) -> float | None:
    """Estimate a robust upper depth limit that removes rare sensor spikes."""
    arr = np.asarray(depth_meters, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(arr) & (arr > 0)
    if not valid_mask.any():
        return None
    return float(np.percentile(arr[valid_mask], percentile))


def valid_depth_mask(
    depth_meters: np.ndarray,
    *,
    max_depth_m: float | None = None,
    max_depth_percentile: float | None = 99.8,
) -> np.ndarray:
    """Build a valid depth mask with optional robust far-depth clipping."""
    arr = np.asarray(depth_meters, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    if not valid.any():
        return valid

    if max_depth_m is None and max_depth_percentile is not None:
        max_depth_m = robust_depth_upper_bound(arr, valid_mask=valid, percentile=max_depth_percentile)

    if max_depth_m is not None:
        valid &= arr <= float(max_depth_m)
    return valid


def backproject_depth_to_pointcloud(
    depth_meters: np.ndarray,
    intr: dict[str, Any],
    *,
    valid_mask: np.ndarray | None = None,
    return_uv: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map into XYZ points using camera intrinsics."""
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["ppx"])
    cy = float(intr["ppy"])

    height, width = depth_meters.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    z = depth_meters.astype(np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(z) & (z > 0)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)
    selected_points = points[valid_mask].astype(np.float32)
    if not return_uv:
        return selected_points

    uv = np.stack([u, v], axis=-1)[valid_mask].astype(np.int32)
    return selected_points, uv


def pointcloud_colors_from_rgb(rgb: np.ndarray, uv_pixels: np.ndarray) -> np.ndarray:
    """Gather RGB colors for a point cloud from the source image pixels."""
    rgb_uint8 = np.asarray(rgb, dtype=np.uint8)
    return rgb_uint8[uv_pixels[:, 1], uv_pixels[:, 0]]


def _sample_points_for_preview(points: np.ndarray, max_points: int) -> np.ndarray:
    """Down-sample a dense point cloud for fast preview rendering."""
    if points.shape[0] <= max_points:
        return points
    indices = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def _sample_points_and_colors_for_preview(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Down-sample a colored point cloud for fast preview rendering."""
    if points.shape[0] <= max_points:
        return points, colors
    indices = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[indices], colors[indices]


def _set_equal_axis_2d(ax, xs: np.ndarray, ys: np.ndarray) -> None:
    """Set equal scaling for a 2D scatter plot."""
    if xs.size == 0 or ys.size == 0:
        return
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min, 1e-6)
    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_aspect("equal", adjustable="box")


def save_pointcloud_preview(
    points: np.ndarray,
    out_png: Path,
    *,
    title: str,
    empty_message: str,
    max_points: int = 20000,
    colors: np.ndarray | None = None,
) -> None:
    """Save a point-cloud preview.

    If RGB colors are provided, save a more informative two-view preview that
    better reflects the original scene structure.
    """
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    if points.shape[0] == 0:
        plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, empty_message, ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        return

    if colors is not None and len(colors) == len(points):
        pts, preview_colors = _sample_points_and_colors_for_preview(
            points,
            np.asarray(colors, dtype=np.uint8),
            max_points=max_points,
        )
        preview_colors = preview_colors.astype(np.float32) / 255.0
        x = pts[:, 0]
        y = -pts[:, 1]
        z = pts[:, 2]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(x, y, c=preview_colors, s=0.2)
        axes[0].set_xlabel("X (meters)")
        axes[0].set_ylabel("-Y (meters)")
        axes[0].set_title(f"{title} Front View")
        _set_equal_axis_2d(axes[0], x, y)

        axes[1].scatter(x, z, c=preview_colors, s=0.2)
        axes[1].set_xlabel("X (meters)")
        axes[1].set_ylabel("Z (meters)")
        axes[1].set_title(f"{title} Side View")
        _set_equal_axis_2d(axes[1], x, z)

        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    pts = _sample_points_for_preview(points, max_points=max_points)
    x = pts[:, 0]
    z = pts[:, 2]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, z, s=0.2)
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_pointcloud_ply(
    path: Path,
    points: np.ndarray,
    *,
    colors: np.ndarray | None = None,
) -> None:
    """Save a point cloud as an ASCII PLY file for external inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape[0] != points.shape[0]:
            raise ValueError("Point and color counts do not match for PLY export.")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        if colors is not None:
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
        handle.write("end_header\n")

        if colors is None:
            for point in points:
                handle.write(f"{point[0]} {point[1]} {point[2]}\n")
            return

        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def save_depth_vis(depth_meters: np.ndarray, out_png: Path) -> None:
    """Render a percentile-clipped depth visualization."""
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(depth_meters, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)

    if valid.any():
        values = arr[valid]
        vmin = float(np.percentile(values, 2))
        vmax = float(np.percentile(values, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        vis = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        vis[~valid] = 0
    else:
        vis = np.zeros_like(arr, dtype=np.float32)

    plt.figure(figsize=(10, 6))
    plt.imshow(vis)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()
