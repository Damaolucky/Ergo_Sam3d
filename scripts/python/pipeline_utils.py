#!/usr/bin/env python3
"""Shared helpers for the human motion geometry preprocessing pipeline."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


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


def backproject_depth_to_pointcloud(depth_meters: np.ndarray, intr: dict[str, Any]) -> np.ndarray:
    """Back-project a depth map into XYZ points using camera intrinsics."""
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["ppx"])
    cy = float(intr["ppy"])

    height, width = depth_meters.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    z = depth_meters.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)
    return points[valid].astype(np.float32)


def _sample_points_for_preview(points: np.ndarray, max_points: int) -> np.ndarray:
    """Down-sample a dense point cloud for fast preview rendering."""
    if points.shape[0] <= max_points:
        return points
    indices = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def save_pointcloud_preview(
    points: np.ndarray,
    out_png: Path,
    *,
    title: str,
    empty_message: str,
    max_points: int = 20000,
) -> None:
    """Save a lightweight X-Z scatter preview of a point cloud."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if points.shape[0] == 0:
        plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, empty_message, ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
        plt.close()
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


def save_depth_vis(depth_meters: np.ndarray, out_png: Path) -> None:
    """Render a percentile-clipped depth visualization."""
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
