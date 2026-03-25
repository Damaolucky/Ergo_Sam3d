#!/usr/bin/env python3
"""Coarsely align a recovered human mesh to the depth-derived human point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline_utils import ensure_output_roots, resolve_in_outputs, write_json


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


def compute_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centroid, eigenvalues, and a right-handed PCA basis."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1.0
    return centroid, eigenvalues, eigenvectors


def coarse_similarity_align(mesh_vertices: np.ndarray, pointcloud: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply a coarse similarity transform by matching PCA frames and scale."""
    mesh_centroid, mesh_eigvals, mesh_basis = compute_pca(mesh_vertices)
    pc_centroid, pc_eigvals, pc_basis = compute_pca(pointcloud)

    mesh_std = np.sqrt(np.maximum(mesh_eigvals, 1e-8))
    pc_std = np.sqrt(np.maximum(pc_eigvals, 1e-8))
    scale = float(np.median(pc_std / mesh_std))

    rotation = pc_basis @ mesh_basis.T
    if np.linalg.det(rotation) < 0:
        pc_basis[:, -1] *= -1.0
        rotation = pc_basis @ mesh_basis.T

    transformed = (mesh_vertices - mesh_centroid) @ rotation.T
    transformed = transformed * scale + pc_centroid
    translation = pc_centroid - scale * (mesh_centroid @ rotation.T)

    stats = {
        "mesh_centroid_before": mesh_centroid.tolist(),
        "pointcloud_centroid": pc_centroid.tolist(),
        "uniform_scale": scale,
        "rotation_matrix": rotation.tolist(),
        "translation": translation.tolist(),
        "mesh_pca_std_before": mesh_std.tolist(),
        "pointcloud_pca_std": pc_std.tolist(),
    }
    return transformed.astype(np.float32), stats


def save_overlay_preview(mesh_vertices: np.ndarray, pointcloud: np.ndarray, out_png: Path) -> None:
    """Save a simple X-Z overlay preview of the aligned mesh and human point cloud."""
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    pc_sample = pointcloud
    if pointcloud.shape[0] > 20000:
        pc_sample = pointcloud[np.random.choice(pointcloud.shape[0], 20000, replace=False)]

    mesh_sample = mesh_vertices
    if mesh_vertices.shape[0] > 8000:
        mesh_sample = mesh_vertices[np.random.choice(mesh_vertices.shape[0], 8000, replace=False)]

    plt.figure(figsize=(7, 5))
    plt.scatter(pc_sample[:, 0], pc_sample[:, 2], s=0.2, label="human point cloud")
    plt.scatter(mesh_sample[:, 0], mesh_sample[:, 2], s=0.2, label="aligned mesh")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.title("Mesh / Point Cloud Overlay (X-Z)")
    plt.legend(markerscale=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    """Run a coarse mesh-to-pointcloud alignment and save the transform outputs."""
    parser = argparse.ArgumentParser(
        description="Coarsely align a recovered human mesh to the depth-derived human point cloud."
    )
    parser.add_argument(
        "clip_dir",
        help="Clip output folder name or path, e.g. 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4",
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

    aligned_vertices, alignment_stats = coarse_similarity_align(mesh_vertices, pointcloud)

    aligned_mesh_path = clip_dir / "aligned_mesh.obj"
    aligned_vertices_path = clip_dir / "aligned_mesh_vertices.npy"
    overlay_path = clip_dir / "mesh_pointcloud_overlay_preview.png"
    stats_path = clip_dir / "alignment_stats.json"

    save_obj(aligned_mesh_path, aligned_vertices, mesh_faces)
    np.save(aligned_vertices_path, aligned_vertices)
    save_overlay_preview(aligned_vertices, pointcloud, overlay_path)

    mesh_extent_before = (mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)).tolist()
    mesh_extent_after = (aligned_vertices.max(axis=0) - aligned_vertices.min(axis=0)).tolist()
    pointcloud_extent = (pointcloud.max(axis=0) - pointcloud.min(axis=0)).tolist()

    payload = {
        "status": "coarse_alignment_complete",
        "clip_dir": str(clip_dir),
        "inputs": {
            "human_mesh_obj": str(mesh_path),
            "human_pointcloud_npy": str(pointcloud_path),
        },
        "transform": alignment_stats,
        "mesh_extent_before": mesh_extent_before,
        "mesh_extent_after": mesh_extent_after,
        "pointcloud_extent": pointcloud_extent,
        "outputs": {
            "aligned_mesh_obj": str(aligned_mesh_path),
            "aligned_mesh_vertices_npy": str(aligned_vertices_path),
            "mesh_pointcloud_overlay_preview_png": str(overlay_path),
            "alignment_stats_json": str(stats_path),
        },
        "notes": [
            "This is a coarse PCA-based similarity alignment, not a final registration method.",
            "Future work can replace this with ICP, correspondences, or torso/pelvis-aware alignment.",
        ],
        "todo": [
            "Reject background outliers in the human point cloud before alignment.",
            "Use a body-centric anchor such as pelvis or torso instead of whole-shape PCA alone.",
            "Add quantitative alignment error metrics once mesh recovery is verified end-to-end.",
        ],
    }
    write_json(stats_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
