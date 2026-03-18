#!/usr/bin/env python3
"""Run PCA-based geometry analysis on a human-only point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline_utils import ensure_output_roots, resolve_in_outputs, write_json


def resolve_clip_dir(clip_arg: str) -> Path:
    """Resolve a clip output directory directly or from the outputs root."""
    return resolve_in_outputs(clip_arg, expect_dir=True, label="Clip directory")


def compute_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centroid, eigenvalues, and eigenvectors for a point cloud."""
    if points.shape[0] < 3:
        raise ValueError("Not enough points for PCA.")

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return centroid, eigvals[order], eigvecs[:, order]


def save_pca_preview(
    points: np.ndarray,
    centroid: np.ndarray,
    eigvecs: np.ndarray,
    out_png: Path,
    max_points: int = 20000,
) -> None:
    """Save an X-Z preview with the first PCA axis overlaid."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if points.shape[0] == 0:
        plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, "No human points", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        return

    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], size=max_points, replace=False)
        pts = points[indices]
    else:
        pts = points

    x = pts[:, 0]
    z = pts[:, 2]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, z, s=0.2, label="human points")

    pc1 = eigvecs[:, 0]
    x0, z0 = centroid[0], centroid[2]
    dx, dz = pc1[0], pc1[2]

    plt.plot([x0 - dx, x0 + dx], [z0 - dz, z0 + dz], linewidth=2, label="PCA axis 1")
    plt.scatter([x0], [z0], s=30, label="centroid")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.title("Human Point Cloud PCA Preview (X-Z)")
    plt.legend(markerscale=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    """Parse arguments and write PCA-based human geometry diagnostics."""
    parser = argparse.ArgumentParser(
        description="Analyze human point cloud geometry for one clip directory."
    )
    parser.add_argument(
        "clip_dir",
        help="Clip output folder name or path, e.g. 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4",
    )
    args = parser.parse_args()

    ensure_output_roots()

    clip_dir = resolve_clip_dir(args.clip_dir)
    human_pc_path = clip_dir / "human_pointcloud.npy"
    if not human_pc_path.exists():
        raise FileNotFoundError(f"Human point cloud not found: {human_pc_path}")

    points = np.load(human_pc_path)
    if points.shape[0] == 0:
        raise RuntimeError("human_pointcloud.npy contains no points.")

    centroid, eigvals, eigvecs = compute_pca(points)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins
    std_along_axes = np.sqrt(np.maximum(eigvals, 0.0))

    pc1 = eigvecs[:, 0]
    forward_xz = np.array([pc1[0], pc1[2]], dtype=np.float64)
    norm_xz = np.linalg.norm(forward_xz)
    if norm_xz > 1e-8:
        forward_xz = forward_xz / norm_xz
    else:
        forward_xz = np.array([0.0, 1.0], dtype=np.float64)

    yaw_rad = float(np.arctan2(forward_xz[1], forward_xz[0]))
    yaw_deg = float(np.degrees(yaw_rad))

    preview_png = clip_dir / "human_pointcloud_pca_preview.png"
    geometry_json = clip_dir / "human_geometry.json"

    save_pca_preview(points, centroid, eigvecs, preview_png)

    result = {
        "clip_dir": str(clip_dir),
        "num_points": int(points.shape[0]),
        "centroid": centroid.tolist(),
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
        "bbox_extent": extents.tolist(),
        "pca_eigenvalues": eigvals.tolist(),
        "pca_axes": {
            "axis1": eigvecs[:, 0].tolist(),
            "axis2": eigvecs[:, 1].tolist(),
            "axis3": eigvecs[:, 2].tolist(),
        },
        "std_along_pca_axes": std_along_axes.tolist(),
        "coarse_orientation": {
            "forward_xz_unit": forward_xz.tolist(),
            "yaw_radians": yaw_rad,
            "yaw_degrees": yaw_deg,
        },
        "outputs": {
            "human_pointcloud_pca_preview_png": str(preview_png),
            "human_geometry_json": str(geometry_json),
        },
    }
    write_json(geometry_json, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
