#!/usr/bin/env python3
"""Align a recovered human mesh to the depth-derived human point cloud.

This stage now prioritizes physically meaningful constraints for later height
reasoning:

- preserve the camera vertical axis (`Y`) instead of allowing full 3D PCA rotation
- estimate only a yaw rotation in the X-Z plane
- translate using torso-centered X/Z anchors and a lower-body Y anchor
- keep the mesh's native human-height prior by default, with an optional
  user-provided target human height for explicit scale calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_utils import ensure_output_roots, resolve_in_outputs, write_json


PREVIEW_RNG = np.random.default_rng(0)


def resolve_clip_dir(clip_arg: str) -> Path:
    """Resolve a clip output directory directly or from the outputs root."""
    return resolve_in_outputs(clip_arg, expect_dir=True, label="Clip directory")


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a simple OBJ file containing vertices and triangular faces."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                parts = line.strip().split()[1:4]
                face = [int(part.split("/")[0]) - 1 for part in parts]
                faces.append(face)
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a simple OBJ mesh with 1-indexed triangular faces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def robust_percentile_range(values: np.ndarray, low: float, high: float) -> float:
    """Return a robust axis range between two percentiles."""
    lower, upper = np.percentile(values, [low, high])
    return float(upper - lower)


def robust_point_cloud_subset(pointcloud: np.ndarray) -> np.ndarray:
    """Drop the most obvious background outliers before alignment statistics.

    The current human mask can still leak background surfaces. For alignment we
    keep the central depth band and a near-full X range, which is enough to
    stabilize yaw and translation without deleting most of the body.
    """
    z_median = float(np.median(pointcloud[:, 2]))
    z_mad = float(np.median(np.abs(pointcloud[:, 2] - z_median)))
    z_band = max(0.35, 2.5 * z_mad)

    x_low, x_high = np.percentile(pointcloud[:, 0], [1, 99])
    keep = (
        (np.abs(pointcloud[:, 2] - z_median) <= z_band)
        & (pointcloud[:, 0] >= x_low)
        & (pointcloud[:, 0] <= x_high)
    )
    subset = pointcloud[keep]
    if subset.shape[0] < max(1000, pointcloud.shape[0] // 10):
        return pointcloud
    return subset


def torso_subset(points: np.ndarray) -> np.ndarray:
    """Select a torso-heavy subset for yaw and center estimation."""
    y_low, y_high = np.percentile(points[:, 1], [20, 80])
    z_median = float(np.median(points[:, 2]))
    z_mad = float(np.median(np.abs(points[:, 2] - z_median)))
    z_band = max(0.35, 2.5 * z_mad)

    keep = (
        (points[:, 1] >= y_low)
        & (points[:, 1] <= y_high)
        & (np.abs(points[:, 2] - z_median) <= z_band)
    )
    subset = points[keep]
    if subset.shape[0] < max(200, points.shape[0] // 20):
        return points
    return subset


def yaw_from_xz(points: np.ndarray) -> float:
    """Estimate a body yaw angle from torso points in the X-Z plane."""
    pts = torso_subset(points)
    xz = pts[:, [0, 2]] - pts[:, [0, 2]].mean(axis=0, keepdims=True)
    covariance = np.cov(xz, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    return float(np.arctan2(principal[1], principal[0]))


def yaw_rotation_matrix(yaw_radians: float) -> np.ndarray:
    """Return a rotation matrix for a yaw-only rotation around the camera Y axis."""
    c = float(np.cos(yaw_radians))
    s = float(np.sin(yaw_radians))
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def compute_mesh_height(vertices: np.ndarray) -> float:
    """Compute a robust mesh height along the vertical camera axis."""
    return robust_percentile_range(vertices[:, 1], 1, 99)


def choose_scale(mesh_vertices: np.ndarray, target_human_height_m: float | None) -> tuple[float, dict]:
    """Choose a physically meaningful scale for alignment."""
    mesh_native_height = compute_mesh_height(mesh_vertices)
    if mesh_native_height <= 1e-6:
        raise ValueError("Mesh height is degenerate and cannot be used for alignment.")

    if target_human_height_m is not None:
        scale = float(target_human_height_m / mesh_native_height)
        mode = "target_human_height"
    else:
        scale = 1.0
        mode = "mesh_native_height_prior"

    return scale, {
        "mode": mode,
        "mesh_native_height_m": mesh_native_height,
        "target_human_height_m": target_human_height_m,
        "applied_scale": scale,
    }


def align_mesh_height_prior(
    mesh_vertices: np.ndarray,
    pointcloud: np.ndarray,
    *,
    target_human_height_m: float | None,
    bottom_anchor_percentile: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Align the mesh using yaw-only rotation and height-prior translation."""
    pointcloud_subset = robust_point_cloud_subset(pointcloud)

    pc_yaw = yaw_from_xz(pointcloud_subset)
    mesh_yaw = yaw_from_xz(mesh_vertices)

    yaw_delta = pc_yaw - mesh_yaw
    rotation = yaw_rotation_matrix(yaw_delta)
    rotated_mesh = mesh_vertices @ rotation.T

    scale, scale_stats = choose_scale(rotated_mesh, target_human_height_m)
    scaled_mesh = rotated_mesh * scale

    pc_torso = torso_subset(pointcloud_subset)
    mesh_torso = torso_subset(scaled_mesh)

    translation_x = float(np.median(pc_torso[:, 0]) - np.median(mesh_torso[:, 0]))
    translation_z = float(np.median(pc_torso[:, 2]) - np.median(mesh_torso[:, 2]))
    translation_y = float(
        np.percentile(pointcloud_subset[:, 1], bottom_anchor_percentile)
        - np.percentile(scaled_mesh[:, 1], bottom_anchor_percentile)
    )
    translation = np.asarray([translation_x, translation_y, translation_z], dtype=np.float32)

    aligned_mesh = scaled_mesh + translation[None, :]

    observed_height_raw = robust_percentile_range(pointcloud[:, 1], 5, 95)
    observed_height_subset = robust_percentile_range(pointcloud_subset[:, 1], 5, 95)
    aligned_height = compute_mesh_height(aligned_mesh)

    stats = {
        "alignment_method": "height_prior_yaw_only",
        "yaw_radians": yaw_delta,
        "yaw_degrees": float(np.degrees(yaw_delta)),
        "rotation_matrix": rotation.tolist(),
        "translation": translation.tolist(),
        "bottom_anchor_percentile": bottom_anchor_percentile,
        "pointcloud_subset_count": int(pointcloud_subset.shape[0]),
        "pointcloud_torso_count": int(pc_torso.shape[0]),
        "mesh_torso_count": int(mesh_torso.shape[0]),
        "pointcloud_torso_median": np.median(pc_torso, axis=0).tolist(),
        "mesh_torso_median_before_translation": np.median(mesh_torso, axis=0).tolist(),
        "height_reference": {
            **scale_stats,
            "aligned_mesh_height_m": aligned_height,
            "observed_pointcloud_height_raw_m": observed_height_raw,
            "observed_pointcloud_height_subset_m": observed_height_subset,
            "pointcloud_bottom_y_m": float(np.percentile(pointcloud_subset[:, 1], bottom_anchor_percentile)),
            "aligned_mesh_bottom_y_m": float(np.percentile(aligned_mesh[:, 1], bottom_anchor_percentile)),
        },
    }
    return aligned_mesh.astype(np.float32), pointcloud_subset.astype(np.float32), stats


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Down-sample a dense point set for preview rendering."""
    if points.shape[0] <= max_points:
        return points
    indices = PREVIEW_RNG.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def set_equal_axis_2d(ax, xs: np.ndarray, ys: np.ndarray) -> None:
    """Use equal scaling for one 2D preview panel."""
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


def nearest_neighbor_metrics(points: np.ndarray, target: np.ndarray) -> dict[str, float] | None:
    """Compute simple nearest-neighbor overlap metrics when SciPy is available."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None

    distances = cKDTree(target).query(points, k=1)[0]
    return {
        "mean_distance_m": float(distances.mean()),
        "median_distance_m": float(np.median(distances)),
        "p95_distance_m": float(np.percentile(distances, 95)),
    }


def apply_similarity_delta(
    vertices: np.ndarray,
    *,
    yaw_radians: float,
    translation_xyz: np.ndarray,
    log_scale: float = 0.0,
) -> np.ndarray:
    """Apply a small yaw/translation/scale update around the mesh centroid."""
    center = vertices.mean(axis=0, keepdims=True)
    scale = float(np.exp(log_scale))
    rotation = yaw_rotation_matrix(yaw_radians)
    centered = (vertices - center) * scale
    transformed = centered @ rotation.T
    return (transformed + center + translation_xyz[None, :]).astype(np.float32)


def mesh_guided_alignment_subset(
    pointcloud: np.ndarray,
    aligned_mesh: np.ndarray,
    *,
    bbox_padding_xyz: tuple[float, float, float] = (0.12, 0.12, 0.12),
    distance_threshold_m: float = 0.20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep only the point-cloud region that is spatially consistent with the mesh.

    The raw human mask can still include shelf and background surfaces. After the
    initial height-prior alignment, the mesh gives a much better support region
    for deciding which depth points are likely to belong to the human body.
    """
    bbox_padding = np.asarray(bbox_padding_xyz, dtype=np.float32)
    bbox_min = aligned_mesh.min(axis=0) - bbox_padding
    bbox_max = aligned_mesh.max(axis=0) + bbox_padding
    bbox_keep = np.all((pointcloud >= bbox_min) & (pointcloud <= bbox_max), axis=1)
    bbox_subset = pointcloud[bbox_keep]

    stats: dict[str, Any] = {
        "bbox_padding_xyz_m": bbox_padding.astype(float).tolist(),
        "bbox_subset_count": int(bbox_subset.shape[0]),
        "distance_threshold_m": distance_threshold_m,
        "distance_pruned_count": None,
        "method": "mesh_bbox_only",
    }

    min_required = max(1000, pointcloud.shape[0] // 20)
    if bbox_subset.shape[0] < min_required:
        stats["fallback"] = "bbox_subset_too_small"
        return pointcloud, stats

    try:
        from scipy.spatial import cKDTree
    except Exception:
        stats["fallback"] = "scipy_not_available"
        return bbox_subset.astype(np.float32), stats

    distances = cKDTree(aligned_mesh).query(bbox_subset, k=1)[0]
    distance_keep = distances <= distance_threshold_m
    distance_subset = bbox_subset[distance_keep]
    stats["distance_pruned_count"] = int(distance_subset.shape[0])

    if distance_subset.shape[0] < min_required:
        stats["fallback"] = "distance_subset_too_small"
        return bbox_subset.astype(np.float32), stats

    stats["method"] = "mesh_bbox_plus_distance_prune"
    stats["distance_pruned_metrics"] = {
        "mean_distance_m": float(distances[distance_keep].mean()),
        "median_distance_m": float(np.median(distances[distance_keep])),
        "p95_distance_m": float(np.percentile(distances[distance_keep], 95)),
    }
    return distance_subset.astype(np.float32), stats


def maybe_refine_alignment(
    aligned_mesh: np.ndarray,
    pointcloud_subset: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine alignment with a multi-stage bounded partial-Chamfer objective.

    This mirrors the scan-fitting intuition used in SMPL-Fitting: keep only the
    scan region that should match the body, then optimize a thresholded
    bidirectional Chamfer-style objective rather than a pure PCA heuristic. The
    thresholds are tightened over stages, similar to a coarse-to-fine fitting
    schedule. Only yaw, translation, and a small global scale correction are
    optimized here; the mesh pose itself is not modified.
    """
    try:
        from scipy.spatial import cKDTree
        from scipy.optimize import minimize
    except Exception:
        return aligned_mesh, {"status": "skipped", "reason": "scipy_not_available"}

    subset_sample = sample_points(pointcloud_subset, 20000)
    initial_metrics = {
        "mesh_to_subset": nearest_neighbor_metrics(aligned_mesh, subset_sample),
        "subset_to_mesh": nearest_neighbor_metrics(subset_sample, aligned_mesh),
    }
    if initial_metrics["mesh_to_subset"] is None or initial_metrics["subset_to_mesh"] is None:
        return aligned_mesh, {"status": "skipped", "reason": "metrics_unavailable"}

    stages = [
        {
            "name": "coarse",
            "mesh_threshold_m": 0.35,
            "scan_threshold_m": 0.30,
            "scan_weight": 0.55,
            "bounds": (
                (-np.deg2rad(35.0), np.deg2rad(35.0)),
                (-0.25, 0.25),
                (-0.18, 0.18),
                (-0.25, 0.25),
                (-0.04, 0.04),
            ),
            "maxiter": 90,
        },
        {
            "name": "medium",
            "mesh_threshold_m": 0.25,
            "scan_threshold_m": 0.20,
            "scan_weight": 0.70,
            "bounds": (
                (-np.deg2rad(18.0), np.deg2rad(18.0)),
                (-0.12, 0.12),
                (-0.10, 0.10),
                (-0.12, 0.12),
                (-0.015, 0.015),
            ),
            "maxiter": 70,
        },
        {
            "name": "fine",
            "mesh_threshold_m": 0.16,
            "scan_threshold_m": 0.14,
            "scan_weight": 0.85,
            "bounds": (
                (-np.deg2rad(8.0), np.deg2rad(8.0)),
                (-0.06, 0.06),
                (-0.06, 0.06),
                (-0.06, 0.06),
                (-0.008, 0.008),
            ),
            "maxiter": 60,
        },
    ]

    current_mesh = aligned_mesh
    accepted_any = False
    stage_stats: list[dict[str, Any]] = []
    cumulative_scale = 1.0
    cumulative_translation = np.zeros(3, dtype=np.float64)
    cumulative_yaw = 0.0

    for stage in stages:
        subset_tree = cKDTree(subset_sample)

        def objective(params: np.ndarray) -> float:
            candidate = apply_similarity_delta(
                current_mesh,
                yaw_radians=float(params[0]),
                translation_xyz=np.asarray(params[1:4], dtype=np.float32),
                log_scale=float(params[4]),
            )
            mesh_to_subset = subset_tree.query(candidate, k=1)[0]
            subset_to_mesh = cKDTree(candidate).query(subset_sample, k=1)[0]
            total_log_scale = float(np.log(cumulative_scale) + params[4])
            scale_regularizer = 8.0 * (total_log_scale ** 2)
            translation_regularizer = 0.03 * float(np.linalg.norm(params[1:4]))
            return float(
                np.minimum(mesh_to_subset, stage["mesh_threshold_m"]).mean()
                + stage["scan_weight"] * np.minimum(subset_to_mesh, stage["scan_threshold_m"]).mean()
                + scale_regularizer
                + translation_regularizer
            )

        initial_params = np.zeros(5, dtype=np.float64)
        initial_score = objective(initial_params)
        result = minimize(
            objective,
            initial_params,
            method="Powell",
            bounds=stage["bounds"],
            options={"maxiter": stage["maxiter"], "xtol": 5e-4, "ftol": 5e-4},
        )

        candidate = apply_similarity_delta(
            current_mesh,
            yaw_radians=float(result.x[0]),
            translation_xyz=np.asarray(result.x[1:4], dtype=np.float32),
            log_scale=float(result.x[4]),
        )
        final_score = float(result.fun)
        proposed_total_scale = cumulative_scale * float(np.exp(result.x[4]))
        accepted = bool(
            result.success
            and final_score < initial_score - 1e-4
            and 0.94 <= proposed_total_scale <= 1.06
        )
        if accepted:
            current_mesh = candidate
            accepted_any = True
            cumulative_yaw += float(result.x[0])
            cumulative_translation += np.asarray(result.x[1:4], dtype=np.float64)
            cumulative_scale *= float(np.exp(result.x[4]))

        stage_stats.append(
            {
                "name": stage["name"],
                "accepted": accepted,
                "initial_score": float(initial_score),
                "final_score": final_score,
                "optimized_delta": {
                    "yaw_degrees": float(np.degrees(result.x[0])),
                    "translation": np.asarray(result.x[1:4], dtype=float).tolist(),
                    "scale_multiplier": float(np.exp(result.x[4])),
                },
                "thresholds": {
                    "mesh_to_scan_m": stage["mesh_threshold_m"],
                    "scan_to_mesh_m": stage["scan_threshold_m"],
                    "scan_weight": stage["scan_weight"],
                },
            }
        )

    final_metrics = {
        "mesh_to_subset": nearest_neighbor_metrics(current_mesh, subset_sample),
        "subset_to_mesh": nearest_neighbor_metrics(subset_sample, current_mesh),
    }

    return current_mesh, {
        "status": "completed",
        "accepted": accepted_any,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "stages": stage_stats,
        "optimized_delta": {
            "yaw_degrees": float(np.degrees(cumulative_yaw)),
            "translation": cumulative_translation.tolist(),
            "scale_multiplier": cumulative_scale,
        },
    }


def save_overlay_preview(
    aligned_mesh: np.ndarray,
    pointcloud: np.ndarray,
    pointcloud_subset: np.ndarray,
    out_png: Path,
) -> None:
    """Save front/top/side overlay previews of the aligned mesh and point cloud.

    The alignment subset is a visible-surface scan subset, not a complete human
    body. Showing only the top-down X-Z view can make it look non-human, so the
    front view is included first for visual sanity checking.
    """
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    raw_sample = sample_points(pointcloud, 12000)
    subset_sample = sample_points(pointcloud_subset, 12000)
    mesh_sample = sample_points(aligned_mesh, 8000)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].scatter(raw_sample[:, 0], -raw_sample[:, 1], s=0.15, alpha=0.12, label="raw human mask point cloud")
    axes[0].scatter(subset_sample[:, 0], -subset_sample[:, 1], s=0.2, alpha=0.35, label="visible body surface subset")
    axes[0].scatter(mesh_sample[:, 0], -mesh_sample[:, 1], s=0.2, alpha=0.6, label="aligned mesh")
    axes[0].set_xlabel("X (meters)")
    axes[0].set_ylabel("-Y (meters)")
    axes[0].set_title("Front Overlay (X, -Y)")
    set_equal_axis_2d(
        axes[0],
        np.concatenate([raw_sample[:, 0], subset_sample[:, 0], mesh_sample[:, 0]]),
        np.concatenate([-raw_sample[:, 1], -subset_sample[:, 1], -mesh_sample[:, 1]]),
    )
    axes[0].legend(markerscale=8)

    axes[1].scatter(raw_sample[:, 0], raw_sample[:, 2], s=0.15, alpha=0.12, label="raw human mask point cloud")
    axes[1].scatter(subset_sample[:, 0], subset_sample[:, 2], s=0.2, alpha=0.35, label="visible body surface subset")
    axes[1].scatter(mesh_sample[:, 0], mesh_sample[:, 2], s=0.2, alpha=0.6, label="aligned mesh")
    axes[1].set_xlabel("X (meters)")
    axes[1].set_ylabel("Z (meters)")
    axes[1].set_title("Top Overlay (X, Z)")
    set_equal_axis_2d(
        axes[1],
        np.concatenate([raw_sample[:, 0], subset_sample[:, 0], mesh_sample[:, 0]]),
        np.concatenate([raw_sample[:, 2], subset_sample[:, 2], mesh_sample[:, 2]]),
    )

    axes[2].scatter(raw_sample[:, 2], -raw_sample[:, 1], s=0.15, alpha=0.12, label="raw human mask point cloud")
    axes[2].scatter(subset_sample[:, 2], -subset_sample[:, 1], s=0.2, alpha=0.35, label="visible body surface subset")
    axes[2].scatter(mesh_sample[:, 2], -mesh_sample[:, 1], s=0.2, alpha=0.6, label="aligned mesh")
    axes[2].set_xlabel("Z (meters)")
    axes[2].set_ylabel("-Y (meters)")
    axes[2].set_title("Side Overlay (Z, -Y)")
    set_equal_axis_2d(
        axes[2],
        np.concatenate([raw_sample[:, 2], subset_sample[:, 2], mesh_sample[:, 2]]),
        np.concatenate([-raw_sample[:, 1], -subset_sample[:, 1], -mesh_sample[:, 1]]),
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_alignment_subset_preview(pointcloud_subset: np.ndarray, out_png: Path) -> None:
    """Save a front/top preview for the visible body-surface subset."""
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    subset_sample = sample_points(pointcloud_subset, 20000)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(subset_sample[:, 0], -subset_sample[:, 1], s=0.2)
    axes[0].set_xlabel("X (meters)")
    axes[0].set_ylabel("-Y (meters)")
    axes[0].set_title("Visible Body Surface Subset - Front")
    set_equal_axis_2d(axes[0], subset_sample[:, 0], -subset_sample[:, 1])

    axes[1].scatter(subset_sample[:, 0], subset_sample[:, 2], s=0.2)
    axes[1].set_xlabel("X (meters)")
    axes[1].set_ylabel("Z (meters)")
    axes[1].set_title("Visible Body Surface Subset - Top")
    set_equal_axis_2d(axes[1], subset_sample[:, 0], subset_sample[:, 2])

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run a height-prior mesh-to-pointcloud alignment and save the outputs."""
    parser = argparse.ArgumentParser(
        description="Align a recovered human mesh to the depth-derived human point cloud."
    )
    parser.add_argument(
        "clip_dir",
        help="Clip output folder name or path, e.g. 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4",
    )
    parser.add_argument(
        "--target-human-height-m",
        type=float,
        default=None,
        help="Optional known human height in meters. If omitted, keep the mesh's native height prior.",
    )
    parser.add_argument(
        "--bottom-anchor-percentile",
        type=float,
        default=95.0,
        help="Percentile used to align the lower body along camera Y (default: 95).",
    )
    args = parser.parse_args()

    ensure_output_roots()

    clip_dir = resolve_clip_dir(args.clip_dir)
    pointcloud_path = clip_dir / "human_pointcloud.npy"
    mesh_path = clip_dir / "human_mesh.obj"

    if not pointcloud_path.exists():
        raise FileNotFoundError(
            f"Human point cloud not found: {pointcloud_path}. Run the verified human-mask stage first."
        )
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Recovered mesh not found: {mesh_path}. Run scripts/bash/run_human_mesh_recovery.sh first."
        )

    pointcloud = np.load(pointcloud_path)
    if pointcloud.shape[0] < 10:
        raise ValueError("human_pointcloud.npy does not contain enough points for alignment.")

    mesh_vertices, mesh_faces = load_obj(mesh_path)
    if mesh_vertices.shape[0] < 10:
        raise ValueError("human_mesh.obj does not contain enough vertices for alignment.")

    aligned_vertices, pointcloud_subset, alignment_stats = align_mesh_height_prior(
        mesh_vertices,
        pointcloud,
        target_human_height_m=args.target_human_height_m,
        bottom_anchor_percentile=args.bottom_anchor_percentile,
    )
    initial_subset, initial_subset_stats = mesh_guided_alignment_subset(pointcloud, aligned_vertices)
    aligned_vertices, refinement_stats = maybe_refine_alignment(aligned_vertices, initial_subset)
    final_subset, final_subset_stats = mesh_guided_alignment_subset(pointcloud, aligned_vertices)
    final_mesh_height = compute_mesh_height(aligned_vertices)
    alignment_stats["initial_alignment_subset"] = initial_subset_stats
    alignment_stats["refinement"] = refinement_stats
    alignment_stats["final_alignment_subset"] = final_subset_stats
    alignment_stats["height_reference"]["aligned_mesh_height_m"] = final_mesh_height
    alignment_stats["height_reference"]["refined_scale_multiplier"] = refinement_stats.get(
        "optimized_delta",
        {},
    ).get("scale_multiplier", 1.0)
    alignment_stats["overlap_metrics"] = {
        "mesh_to_alignment_subset": nearest_neighbor_metrics(aligned_vertices, final_subset),
        "alignment_subset_to_mesh": nearest_neighbor_metrics(final_subset, aligned_vertices),
    }

    aligned_mesh_path = clip_dir / "aligned_mesh.obj"
    aligned_vertices_path = clip_dir / "aligned_mesh_vertices.npy"
    alignment_subset_path = clip_dir / "alignment_pointcloud_subset.npy"
    alignment_subset_preview_path = clip_dir / "alignment_pointcloud_subset_preview.png"
    overlay_path = clip_dir / "mesh_pointcloud_overlay_preview.png"
    stats_path = clip_dir / "alignment_stats.json"

    save_obj(aligned_mesh_path, aligned_vertices, mesh_faces)
    np.save(aligned_vertices_path, aligned_vertices)
    np.save(alignment_subset_path, final_subset)
    save_alignment_subset_preview(final_subset, alignment_subset_preview_path)
    save_overlay_preview(aligned_vertices, pointcloud, final_subset, overlay_path)

    mesh_extent_before = (mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)).tolist()
    mesh_extent_after = (aligned_vertices.max(axis=0) - aligned_vertices.min(axis=0)).tolist()
    pointcloud_extent = (pointcloud.max(axis=0) - pointcloud.min(axis=0)).tolist()
    pointcloud_subset_extent = (final_subset.max(axis=0) - final_subset.min(axis=0)).tolist()

    payload = {
        "status": "height_prior_alignment_complete",
        "clip_dir": str(clip_dir),
        "inputs": {
            "human_mesh_obj": str(mesh_path),
            "human_pointcloud_npy": str(pointcloud_path),
        },
        "transform": alignment_stats,
        "estimated_human_height_m": final_mesh_height,
        "mesh_extent_before": mesh_extent_before,
        "mesh_extent_after": mesh_extent_after,
        "pointcloud_extent": pointcloud_extent,
        "pointcloud_subset_extent": pointcloud_subset_extent,
        "outputs": {
            "aligned_mesh_obj": str(aligned_mesh_path),
            "aligned_mesh_vertices_npy": str(aligned_vertices_path),
            "alignment_pointcloud_subset_npy": str(alignment_subset_path),
            "alignment_pointcloud_subset_preview_png": str(alignment_subset_preview_path),
            "mesh_pointcloud_overlay_preview_png": str(overlay_path),
            "alignment_stats_json": str(stats_path),
        },
        "notes": [
            "This alignment preserves the camera vertical axis and solves only for yaw, translation, and optional height-based scale.",
            "By default the mesh keeps its native human-height prior; pass --target-human-height-m when the subject's height is known.",
            "The saved alignment point-cloud subset is a mesh-guided visible body-surface scan, not a full human silhouette.",
            "Use the front overlay panel to inspect human shape; the top view can look like a thin orange band for side-view depth.",
        ],
        "todo": [
            "Estimate cabinet geometry in the same depth frame and compare its top height against the aligned human height reference.",
            "Promote the current alignment subset heuristic into an explicit body-only point-cloud cleaning stage shared across the pipeline.",
            "Add quantitative overlap metrics once cabinet and floor landmarks are available.",
        ],
    }
    write_json(stats_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
