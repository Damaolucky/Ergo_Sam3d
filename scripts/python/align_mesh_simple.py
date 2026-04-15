#!/usr/bin/env python3
"""Simple rigid alignment: scale the recovered mesh to match the depth point cloud.

Does NOT change pose or body shape — only applies uniform scale + translation
so the mesh has metric measurements aligned to the depth surface.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load_obj(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith("f "):
                face = [int(tok.split("/")[0]) - 1 for tok in line.strip().split()[1:]]
                faces.append(face)
    return np.array(vertices, dtype=np.float32), faces


def save_obj(path: Path, vertices: np.ndarray, faces: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def mesh_height(vertices: np.ndarray) -> float:
    """Mesh height along Y axis (camera Y points down)."""
    return float(vertices[:, 1].max() - vertices[:, 1].min())


def find_scale_by_chamfer(
    mesh_verts: np.ndarray,
    pc_points: np.ndarray,
    scale_range: tuple[float, float] = (0.005, 0.05),
    n_steps: int = 200,
) -> tuple[float, float]:
    """Find uniform scale that minimizes chamfer distance from mesh to point cloud.

    The mesh from HMR/TokenHMR is in an arbitrary scale (roughly -1 to 1 range).
    The point cloud is in meters. We search for the scale that best matches.
    """
    # Center both
    mesh_center = mesh_verts.mean(axis=0)
    pc_center = pc_points.mean(axis=0)
    mesh_centered = mesh_verts - mesh_center

    pc_tree = cKDTree(pc_points)

    best_scale = None
    best_score = float("inf")

    scales = np.linspace(scale_range[0], scale_range[1], n_steps)
    for s in scales:
        scaled = mesh_centered * s + pc_center
        # Chamfer: mesh→pc distance
        dists, _ = pc_tree.query(scaled, k=1)
        # Use median to be robust to outliers (limbs not visible in depth)
        score = float(np.median(dists))
        if score < best_score:
            best_score = score
            best_scale = s

    return best_scale, best_score


def align_mesh_to_pointcloud(
    mesh_verts: np.ndarray,
    pc_points: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Align mesh to point cloud with uniform scale + translation only.

    The mesh from HMR/TokenHMR is already roughly in meters but centered near
    the origin.  The point cloud is in depth camera coordinates (meters).

    Steps:
    1. Match heights: scale = pc_height / mesh_height
    2. Translate: anchor feet (bottom Y), match torso X/Z
    3. Fine-tune scale by searching around the height-based estimate
    """
    mesh_h = mesh_height(mesh_verts)
    pc_h = float(np.percentile(pc_points[:, 1], 95) - np.percentile(pc_points[:, 1], 5))

    # Height-based scale (mesh is already in meters, so scale ~ 1.0)
    height_scale = pc_h / mesh_h if mesh_h > 0 else 1.0

    # Search for best scale around the height estimate
    pc_tree = cKDTree(pc_points)

    def score_alignment(scale: float) -> tuple[float, np.ndarray]:
        """Score a scale by: scale mesh, translate to match pc, measure chamfer."""
        mesh_center = mesh_verts.mean(axis=0)
        scaled = (mesh_verts - mesh_center) * scale

        # Anchor feet: match 95th percentile Y
        pc_bottom = float(np.percentile(pc_points[:, 1], 95))
        mesh_bottom = float(np.percentile(scaled[:, 1], 95))
        scaled[:, 1] += pc_bottom - mesh_bottom

        # Match torso X, Z
        sy_min, sy_max = scaled[:, 1].min(), scaled[:, 1].max()
        torso_lo = sy_min + 0.3 * (sy_max - sy_min)
        torso_hi = sy_min + 0.7 * (sy_max - sy_min)
        m_torso = scaled[(scaled[:, 1] > torso_lo) & (scaled[:, 1] < torso_hi)]
        p_torso = pc_points[(pc_points[:, 1] > torso_lo) & (pc_points[:, 1] < torso_hi)]
        if len(m_torso) > 10 and len(p_torso) > 10:
            xz_shift = np.median(p_torso[:, [0, 2]], axis=0) - np.median(m_torso[:, [0, 2]], axis=0)
            scaled[:, 0] += xz_shift[0]
            scaled[:, 2] += xz_shift[1]

        dists, _ = pc_tree.query(scaled, k=1)
        return float(np.median(dists)), scaled

    best_scale = height_scale
    best_score = float("inf")
    best_aligned = None

    for s in np.linspace(height_scale * 0.8, height_scale * 1.2, 200):
        score, aligned = score_alignment(s)
        if score < best_score:
            best_score = score
            best_scale = s
            best_aligned = aligned

    aligned = best_aligned

    # Final metrics
    dists_mesh_to_pc, _ = pc_tree.query(aligned, k=1)
    mesh_tree = cKDTree(aligned)
    dists_pc_to_mesh, _ = mesh_tree.query(pc_points, k=1)

    stats = {
        "method": "uniform_scale_translation_only",
        "height_based_scale": float(height_scale),
        "optimized_scale": float(best_scale),
        "mesh_height_before": float(mesh_h),
        "mesh_height_after_m": float(mesh_height(aligned)),
        "pointcloud_height_m": float(pc_h),
        "chamfer_median_m": float(best_score),
        "mesh_to_pc": {
            "mean_m": float(dists_mesh_to_pc.mean()),
            "median_m": float(np.median(dists_mesh_to_pc)),
            "p95_m": float(np.percentile(dists_mesh_to_pc, 95)),
        },
        "pc_to_mesh": {
            "mean_m": float(dists_pc_to_mesh.mean()),
            "median_m": float(np.median(dists_pc_to_mesh)),
            "p95_m": float(np.percentile(dists_pc_to_mesh, 95)),
        },
    }
    return aligned.astype(np.float32), stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple rigid alignment: uniform scale + translation, no pose/shape change."
    )
    parser.add_argument("clip_dir", help="Clip output folder")
    parser.add_argument("--mesh", default="human_mesh.obj",
                        help="Mesh filename in clip_dir (default: human_mesh.obj)")
    parser.add_argument("--output-prefix", default="aligned_simple",
                        help="Output filename prefix (default: aligned_simple)")
    args = parser.parse_args()

    work_root = Path(os.environ.get("ERGO_WORK_ROOT", str(Path.home() / "hzhou")))
    outputs_dir = work_root / "outputs"

    clip_dir = Path(args.clip_dir)
    if not clip_dir.is_dir():
        clip_dir = outputs_dir / args.clip_dir
    if not clip_dir.is_dir():
        raise FileNotFoundError(f"Clip directory not found: {args.clip_dir}")

    mesh_path = clip_dir / args.mesh
    pc_path = clip_dir / "human_pointcloud.npy"

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if not pc_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {pc_path}")

    mesh_verts, mesh_faces = load_obj(mesh_path)
    pc_points = np.load(pc_path)

    print(f"mesh: {mesh_verts.shape[0]} vertices")
    print(f"pointcloud: {pc_points.shape[0]} points")

    aligned, stats = align_mesh_to_pointcloud(mesh_verts, pc_points)

    out_obj = clip_dir / f"{args.output_prefix}.obj"
    out_npy = clip_dir / f"{args.output_prefix}_vertices.npy"
    out_json = clip_dir / f"{args.output_prefix}_stats.json"

    save_obj(out_obj, aligned, mesh_faces)
    np.save(out_npy, aligned)

    stats["inputs"] = {
        "mesh": str(mesh_path),
        "pointcloud": str(pc_path),
    }
    stats["outputs"] = {
        "aligned_obj": str(out_obj),
        "aligned_vertices_npy": str(out_npy),
    }
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
