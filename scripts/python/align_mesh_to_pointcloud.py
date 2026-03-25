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


def maybe_refine_alignment(
    aligned_mesh: np.ndarray,
    pointcloud_subset: np.ndarray,
    *,
    max_iterations: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a conservative nearest-neighbor refinement and keep it only if it improves."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return aligned_mesh, {"status": "skipped", "reason": "scipy_not_available"}

    rng = np.random.default_rng(0)
    tree = cKDTree(pointcloud_subset)
    best_mesh = aligned_mesh.copy()
    initial_metrics = nearest_neighbor_metrics(best_mesh, pointcloud_subset)
    best_metrics = initial_metrics
    if best_metrics is None:
        return aligned_mesh, {"status": "skipped", "reason": "metrics_unavailable"}

    best_score = best_metrics["mean_distance_m"] + 0.25 * best_metrics["p95_distance_m"]
    accepted_steps: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        sample_indices = rng.choice(best_mesh.shape[0], size=min(4000, best_mesh.shape[0]), replace=False)
        sample = best_mesh[sample_indices]
        distances, matched_idx = tree.query(sample, k=1)
        keep = distances <= min(0.25, float(np.percentile(distances, 75)))
        if int(keep.sum()) < 100:
            break

        mesh_matches = sample[keep]
        point_matches = pointcloud_subset[matched_idx[keep]]

        mesh_xz = mesh_matches[:, [0, 2]] - mesh_matches[:, [0, 2]].mean(axis=0, keepdims=True)
        point_xz = point_matches[:, [0, 2]] - point_matches[:, [0, 2]].mean(axis=0, keepdims=True)
        covariance = mesh_xz.T @ point_xz
        u_mat, _, vt_mat = np.linalg.svd(covariance)
        rotation_2d = vt_mat.T @ u_mat.T
        if np.linalg.det(rotation_2d) < 0:
            vt_mat[-1, :] *= -1.0
            rotation_2d = vt_mat.T @ u_mat.T

        yaw_delta = float(np.arctan2(rotation_2d[1, 0], rotation_2d[0, 0]))
        yaw_delta = float(np.clip(yaw_delta, np.deg2rad(-5.0), np.deg2rad(5.0)))
        rotation = yaw_rotation_matrix(yaw_delta)
        candidate = best_mesh @ rotation.T

        candidate_sample = candidate[sample_indices]
        distances, matched_idx = tree.query(candidate_sample, k=1)
        keep = distances <= min(0.20, float(np.percentile(distances, 70)))
        if int(keep.sum()) < 100:
            break

        delta = pointcloud_subset[matched_idx[keep]] - candidate_sample[keep]
        translation = np.median(delta, axis=0)
        translation = np.clip(translation, [-0.05, -0.03, -0.05], [0.05, 0.03, 0.05])
        candidate = candidate + translation[None, :]

        candidate_metrics = nearest_neighbor_metrics(candidate, pointcloud_subset)
        if candidate_metrics is None:
            break
        candidate_score = (
            candidate_metrics["mean_distance_m"] + 0.25 * candidate_metrics["p95_distance_m"]
        )

        if candidate_score + 1e-6 < best_score:
            best_mesh = candidate.astype(np.float32)
            best_metrics = candidate_metrics
            best_score = candidate_score
            accepted_steps.append(
                {
                    "iteration": iteration,
                    "accepted_yaw_degrees": float(np.degrees(yaw_delta)),
                    "accepted_translation": translation.astype(float).tolist(),
                    "metrics": candidate_metrics,
                }
            )
        else:
            break

    return best_mesh, {
        "status": "completed",
        "accepted_iterations": len(accepted_steps),
        "initial_metrics": initial_metrics,
        "accepted_steps": accepted_steps,
        "final_metrics": nearest_neighbor_metrics(best_mesh, pointcloud_subset),
    }


def save_overlay_preview(
    aligned_mesh: np.ndarray,
    pointcloud: np.ndarray,
    pointcloud_subset: np.ndarray,
    out_png: Path,
) -> None:
    """Save X-Z and Y-Z overlay previews of the aligned mesh and point cloud."""
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    raw_sample = sample_points(pointcloud, 12000)
    subset_sample = sample_points(pointcloud_subset, 12000)
    mesh_sample = sample_points(aligned_mesh, 8000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(raw_sample[:, 0], raw_sample[:, 2], s=0.15, alpha=0.12, label="raw point cloud")
    axes[0].scatter(subset_sample[:, 0], subset_sample[:, 2], s=0.2, alpha=0.35, label="alignment subset")
    axes[0].scatter(mesh_sample[:, 0], mesh_sample[:, 2], s=0.2, alpha=0.6, label="aligned mesh")
    axes[0].set_xlabel("X (meters)")
    axes[0].set_ylabel("Z (meters)")
    axes[0].set_title("Overlay (X-Z)")
    axes[0].legend(markerscale=8)

    axes[1].scatter(raw_sample[:, 2], raw_sample[:, 1], s=0.15, alpha=0.12, label="raw point cloud")
    axes[1].scatter(subset_sample[:, 2], subset_sample[:, 1], s=0.2, alpha=0.35, label="alignment subset")
    axes[1].scatter(mesh_sample[:, 2], mesh_sample[:, 1], s=0.2, alpha=0.6, label="aligned mesh")
    axes[1].set_xlabel("Z (meters)")
    axes[1].set_ylabel("Y (meters)")
    axes[1].set_title("Overlay (Z-Y)")
    axes[1].legend(markerscale=8)

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
    aligned_vertices, refinement_stats = maybe_refine_alignment(aligned_vertices, pointcloud_subset)
    alignment_stats["refinement"] = refinement_stats

    aligned_mesh_path = clip_dir / "aligned_mesh.obj"
    aligned_vertices_path = clip_dir / "aligned_mesh_vertices.npy"
    overlay_path = clip_dir / "mesh_pointcloud_overlay_preview.png"
    stats_path = clip_dir / "alignment_stats.json"

    save_obj(aligned_mesh_path, aligned_vertices, mesh_faces)
    np.save(aligned_vertices_path, aligned_vertices)
    save_overlay_preview(aligned_vertices, pointcloud, pointcloud_subset, overlay_path)

    mesh_extent_before = (mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)).tolist()
    mesh_extent_after = (aligned_vertices.max(axis=0) - aligned_vertices.min(axis=0)).tolist()
    pointcloud_extent = (pointcloud.max(axis=0) - pointcloud.min(axis=0)).tolist()
    pointcloud_subset_extent = (pointcloud_subset.max(axis=0) - pointcloud_subset.min(axis=0)).tolist()

    payload = {
        "status": "height_prior_alignment_complete",
        "clip_dir": str(clip_dir),
        "inputs": {
            "human_mesh_obj": str(mesh_path),
            "human_pointcloud_npy": str(pointcloud_path),
        },
        "transform": alignment_stats,
        "mesh_extent_before": mesh_extent_before,
        "mesh_extent_after": mesh_extent_after,
        "pointcloud_extent": pointcloud_extent,
        "pointcloud_subset_extent": pointcloud_subset_extent,
        "outputs": {
            "aligned_mesh_obj": str(aligned_mesh_path),
            "aligned_mesh_vertices_npy": str(aligned_vertices_path),
            "mesh_pointcloud_overlay_preview_png": str(overlay_path),
            "alignment_stats_json": str(stats_path),
        },
        "notes": [
            "This alignment preserves the camera vertical axis and solves only for yaw, translation, and optional height-based scale.",
            "By default the mesh keeps its native human-height prior; pass --target-human-height-m when the subject's height is known.",
            "The point-cloud subset is only used to stabilize alignment against mask leakage and background contamination.",
        ],
        "todo": [
            "Estimate cabinet geometry in the same depth frame and compare its top height against the aligned human height reference.",
            "Replace the current subset heuristic with a more explicit body-only point-cloud cleaning stage.",
            "Add quantitative overlap metrics once cabinet and floor landmarks are available.",
        ],
    }
    write_json(stats_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
