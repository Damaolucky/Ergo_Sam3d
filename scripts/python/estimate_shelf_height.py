#!/usr/bin/env python3
"""Estimate a shelf/object height from the prepared keyframe sample.

The prepared keyframe is action-aware: `*_lift.mp4` clips use the first frame at
the source shelf/object position, while `*_put.mp4` clips use the last frame at
the destination shelf/object position. In the camera frame used by the depth
backprojection, `Y` points downward, so a vertical height above the floor is
computed as:

    height_m = floor_y_m - target_y_m

The floor reference is taken from the aligned human mesh feet when available.
The target shelf/object height is estimated from a shelf-side region of the
prepared RGB/depth keyframe and stored with an uncertainty band for manual
review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline_utils import (
    backproject_depth_to_pointcloud,
    ensure_output_roots,
    load_json,
    load_pickle,
    resolve_in_outputs,
    resize_rgb_to_shape,
    valid_depth_mask,
    write_json,
)


LEVEL_CONFIGS = {
    # Fractions of image height, plus the default height percentile to use.
    # Smaller image v / camera Y means physically higher in the scene.
    "high": {"v_fraction": (0.10, 0.36), "height_percentile": 85.0},
    "mid": {"v_fraction": (0.30, 0.62), "height_percentile": 55.0},
    "low": {"v_fraction": (0.52, 0.92), "height_percentile": 25.0},
}


def resolve_clip_dir(clip_arg: str) -> Path:
    """Resolve a clip output directory directly or from the outputs root."""
    return resolve_in_outputs(clip_arg, expect_dir=True, label="Clip directory")


def find_one(clip_dir: Path, suffix: str) -> Path:
    """Find one sample artifact by suffix inside a clip directory."""
    preferred = clip_dir / f"{clip_dir.name}{suffix}"
    if preferred.exists():
        return preferred
    matches = sorted(clip_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No artifact ending with {suffix} found in {clip_dir}")
    return matches[0]


def load_sample_metadata(clip_dir: Path) -> dict[str, Any]:
    """Load the sample manifest when present, otherwise return a minimal payload."""
    matches = sorted(clip_dir.glob("*.sample_manifest.json"))
    if not matches:
        return {
            "sample_name": clip_dir.name,
            "sample_role": None,
            "position_label": None,
        }
    return load_json(matches[0])


def infer_level(position_label: str | None, explicit_level: str | None) -> str:
    """Infer the shelf level from metadata unless the caller overrides it."""
    if explicit_level:
        return explicit_level
    label = (position_label or "").lower()
    for level in ("high", "mid", "low"):
        if level in label:
            return level
    return "mid"


def human_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the human mask bbox as xmin, ymin, xmax, ymax."""
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def side_mask_for_frame(
    *,
    width: int,
    human_box: tuple[int, int, int, int] | None,
    side: str,
    margin_pixels: int,
) -> np.ndarray:
    """Build a horizontal image-side mask for the target shelf/object region."""
    columns = np.arange(width)[None, :]
    if human_box is None:
        if side == "left":
            return columns < int(width * 0.48)
        return columns > int(width * 0.52)

    xmin, _, xmax, _ = human_box
    if side == "left":
        limit = min(xmin - margin_pixels, int(width * 0.48))
        return columns < limit
    limit = max(xmax + margin_pixels, int(width * 0.52))
    return columns > limit


def choose_auto_side(
    valid: np.ndarray,
    human_mask: np.ndarray,
    human_box: tuple[int, int, int, int] | None,
    *,
    v_range: tuple[int, int],
    margin_pixels: int,
) -> str:
    """Choose the side with the stronger target-band depth support."""
    height, width = valid.shape
    rows = np.arange(height)[:, None]
    band = (rows >= v_range[0]) & (rows <= v_range[1])
    scores: dict[str, int] = {}
    for side in ("left", "right"):
        side_mask = side_mask_for_frame(
            width=width,
            human_box=human_box,
            side=side,
            margin_pixels=margin_pixels,
        )
        scores[side] = int((valid & ~human_mask & band & side_mask).sum())
    return "right" if scores["right"] >= scores["left"] else "left"


def load_floor_reference(clip_dir: Path, fallback_points: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Estimate floor Y from the aligned human mesh feet when available."""
    aligned_vertices_path = clip_dir / "aligned_mesh_vertices.npy"
    if aligned_vertices_path.exists():
        vertices = np.load(aligned_vertices_path)
        floor_y = float(np.percentile(vertices[:, 1], 99.0))
        return floor_y, {
            "method": "aligned_mesh_feet_y_p99",
            "source": str(aligned_vertices_path),
            "floor_y_m": floor_y,
        }

    floor_y = float(np.percentile(fallback_points[:, 1], 95.0))
    return floor_y, {
        "method": "candidate_scene_y_p95_fallback",
        "source": "candidate_points",
        "floor_y_m": floor_y,
    }


def load_human_height(clip_dir: Path) -> float | None:
    """Load the current aligned human height estimate when available."""
    stats_path = clip_dir / "alignment_stats.json"
    if not stats_path.exists():
        return None
    stats = load_json(stats_path)
    height = stats.get("estimated_human_height_m")
    return float(height) if height is not None else None


def maybe_open3d_filter(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Optionally denoise target candidates with Open3D if it is installed."""
    stats: dict[str, Any] = {"available": False, "used": False}
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        stats["reason"] = f"open3d_not_available: {exc.__class__.__name__}"
        return points, stats

    stats["available"] = True
    if points.shape[0] < 100:
        stats["reason"] = "too_few_points"
        return points, stats

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.015)
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered = np.asarray(point_cloud.points, dtype=np.float32)
    if filtered.shape[0] < 100:
        stats["reason"] = "filtered_too_few_points"
        return points, stats

    stats.update(
        {
            "used": True,
            "input_points": int(points.shape[0]),
            "filtered_points": int(filtered.shape[0]),
        }
    )
    return filtered, stats


def select_target_height(
    heights: np.ndarray,
    *,
    level: str,
    fallback_percentile: float,
) -> tuple[float, dict[str, Any]]:
    """Select a target height from dense shelf/object surface modes."""
    heights = np.asarray(heights, dtype=np.float32)
    fallback = float(np.percentile(heights, fallback_percentile))
    quantiles = np.percentile(heights, [10, 35, 50, 65, 90])

    if level == "high":
        keep_low, keep_high = float(quantiles[3]), float(quantiles[4])
    elif level == "low":
        keep_low, keep_high = float(quantiles[0]), float(quantiles[1])
    else:
        keep_low, keep_high = float(quantiles[1]), float(quantiles[3])

    hist_min = max(0.0, float(heights.min()) - 0.03)
    hist_max = min(3.2, float(heights.max()) + 0.03)
    bin_edges = np.arange(hist_min, hist_max + 0.0301, 0.03)
    if bin_edges.size < 5:
        return fallback, {
            "method": "fallback_percentile_too_few_bins",
            "fallback_percentile": fallback_percentile,
            "fallback_height_m": fallback,
        }

    counts, edges = np.histogram(heights, bins=bin_edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    smooth = np.convolve(counts.astype(np.float32), np.ones(5, dtype=np.float32) / 5.0, mode="same")
    min_peak_count = max(100.0, 0.05 * float(smooth.max(initial=0.0)))

    peaks: list[dict[str, Any]] = []
    for index in range(1, smooth.shape[0] - 1):
        if smooth[index] < min_peak_count:
            continue
        if smooth[index] >= smooth[index - 1] and smooth[index] >= smooth[index + 1]:
            height = float(centers[index])
            peaks.append(
                {
                    "height_m": height,
                    "smoothed_count": float(smooth[index]),
                    "raw_count": int(counts[index]),
                    "in_level_band": bool(keep_low <= height <= keep_high),
                }
            )

    level_peaks = [peak for peak in peaks if peak["in_level_band"]]
    if not level_peaks:
        return fallback, {
            "method": "fallback_percentile_no_level_peak",
            "fallback_percentile": fallback_percentile,
            "fallback_height_m": fallback,
            "level_height_band_m": [keep_low, keep_high],
            "peaks": peaks[:12],
        }

    selected = max(level_peaks, key=lambda peak: peak["smoothed_count"])
    return float(selected["height_m"]), {
        "method": "level_band_histogram_mode",
        "fallback_percentile": fallback_percentile,
        "fallback_height_m": fallback,
        "level_height_band_m": [keep_low, keep_high],
        "selected_peak": selected,
        "peaks": sorted(peaks, key=lambda peak: peak["smoothed_count"], reverse=True)[:12],
    }


def save_height_preview(
    *,
    rgb: np.ndarray,
    candidate_uv: np.ndarray,
    selected_uv: np.ndarray,
    v_range: tuple[int, int],
    side: str,
    out_path: Path,
) -> None:
    """Save an RGB overlay showing the shelf-height candidate region."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(rgb)
    ax.axhspan(v_range[0], v_range[1], color="yellow", alpha=0.12, label="target level band")

    if candidate_uv.size:
        candidate_sample = candidate_uv
        if candidate_sample.shape[0] > 4000:
            idx = np.random.default_rng(0).choice(candidate_sample.shape[0], 4000, replace=False)
            candidate_sample = candidate_sample[idx]
        ax.scatter(candidate_sample[:, 0], candidate_sample[:, 1], s=1, c="cyan", alpha=0.12, label="candidates")

    if selected_uv.size:
        selected_sample = selected_uv
        if selected_sample.shape[0] > 2500:
            idx = np.random.default_rng(1).choice(selected_sample.shape[0], 2500, replace=False)
            selected_sample = selected_sample[idx]
        ax.scatter(selected_sample[:, 0], selected_sample[:, 1], s=2, c="red", alpha=0.35, label="height band")

    ax.set_title(f"Shelf/object height candidates ({side} side)")
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.axis("off")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_height_report(
    *,
    rgb: np.ndarray,
    candidate_uv: np.ndarray,
    selected_uv: np.ndarray,
    filtered_heights: np.ndarray,
    v_range: tuple[int, int],
    side: str,
    target_height_m: float,
    uncertainty_band_m: tuple[float, float],
    height_selection_stats: dict[str, Any],
    human_height_m: float | None,
    height_ratio: float | None,
    calibrated_height_m: float | None,
    out_path: Path,
) -> None:
    """Save a combined visual report for manual shelf-height inspection."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(rgb)
    axes[0].axhspan(v_range[0], v_range[1], color="yellow", alpha=0.12, label="target level band")
    if candidate_uv.size:
        candidate_sample = candidate_uv
        if candidate_sample.shape[0] > 3500:
            idx = np.random.default_rng(0).choice(candidate_sample.shape[0], 3500, replace=False)
            candidate_sample = candidate_sample[idx]
        axes[0].scatter(candidate_sample[:, 0], candidate_sample[:, 1], s=1, c="cyan", alpha=0.12, label="candidates")
    if selected_uv.size:
        selected_sample = selected_uv
        if selected_sample.shape[0] > 2500:
            idx = np.random.default_rng(1).choice(selected_sample.shape[0], 2500, replace=False)
            selected_sample = selected_sample[idx]
        axes[0].scatter(selected_sample[:, 0], selected_sample[:, 1], s=2, c="red", alpha=0.35, label="selected height band")
    axes[0].set_title(f"Keyframe target ROI ({side} side)")
    axes[0].set_xlim(0, rgb.shape[1])
    axes[0].set_ylim(rgb.shape[0], 0)
    axes[0].axis("off")
    axes[0].legend(loc="lower left")

    axes[1].hist(filtered_heights, bins=80, color="#8fa7b3", alpha=0.8, label="candidate heights")
    axes[1].axvspan(
        uncertainty_band_m[0],
        uncertainty_band_m[1],
        color="red",
        alpha=0.16,
        label="selected uncertainty band",
    )
    axes[1].axvline(target_height_m, color="red", linewidth=2.0, label="selected height")
    fallback_height = height_selection_stats.get("fallback_height_m")
    if fallback_height is not None:
        axes[1].axvline(
            float(fallback_height),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="fallback percentile",
        )
    axes[1].set_xlabel("height above floor (meters)")
    axes[1].set_ylabel("candidate count")
    axes[1].set_title("Depth ROI height distribution")
    axes[1].legend(loc="upper right")

    lines = [
        f"estimated shelf/object height: {target_height_m:.3f} m",
        f"uncertainty band: [{uncertainty_band_m[0]:.3f}, {uncertainty_band_m[1]:.3f}] m",
    ]
    if human_height_m is not None:
        lines.append(f"estimated human height: {human_height_m:.3f} m")
    if height_ratio is not None:
        lines.append(f"shelf / human ratio: {height_ratio:.3f}")
    if calibrated_height_m is not None:
        lines.append(f"known-height calibrated shelf height: {calibrated_height_m:.3f} m")
    lines.append(f"selection method: {height_selection_stats.get('method', 'unknown')}")
    axes[1].text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_height_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a concise text summary of the shelf-height estimate."""
    calibrated = payload.get("known_human_height_calibrated_shelf_height_m")
    lines = [
        "Shelf/object height estimate",
        f"clip_dir: {payload['clip_dir']}",
        f"position_label: {payload.get('position_label')}",
        f"target_level: {payload.get('target_level')}",
        f"shelf_side: {payload.get('shelf_side')}",
        f"estimated_shelf_height_m: {payload['estimated_shelf_height_m']:.4f}",
        (
            "estimated_shelf_height_uncertainty_band_m: "
            f"[{payload['estimated_shelf_height_uncertainty_band_m'][0]:.4f}, "
            f"{payload['estimated_shelf_height_uncertainty_band_m'][1]:.4f}]"
        ),
        f"estimated_human_height_m: {payload.get('estimated_human_height_m')}",
        f"shelf_to_human_height_ratio: {payload.get('shelf_to_human_height_ratio')}",
        f"known_human_height_calibrated_shelf_height_m: {calibrated}",
        f"method: {payload.get('method')}",
        "view: shelf_height_report.png and shelf_height_preview.png",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Estimate the prepared-keyframe target shelf/object height."""
    parser = argparse.ArgumentParser(
        description="Estimate the shelf/object height from a prepared keyframe clip output folder."
    )
    parser.add_argument("clip_dir", help="Clip output folder name or path.")
    parser.add_argument(
        "--level",
        choices=["high", "mid", "low"],
        default=None,
        help="Override the target level. Defaults to the position label in the sample manifest.",
    )
    parser.add_argument(
        "--shelf-side",
        choices=["left", "right", "auto"],
        default="right",
        help="Which image side contains the shelf/object target. Default: right.",
    )
    parser.add_argument(
        "--height-percentile",
        type=float,
        default=None,
        help="Override the percentile of target-region heights used for the estimate.",
    )
    parser.add_argument(
        "--known-human-height-m",
        type=float,
        default=None,
        help="Optional real human height. If provided, also report shelf height scaled by this reference.",
    )
    parser.add_argument(
        "--margin-pixels",
        type=int,
        default=24,
        help="Horizontal margin away from the human mask bbox before searching shelf-side points.",
    )
    args = parser.parse_args()

    ensure_output_roots()
    clip_dir = resolve_clip_dir(args.clip_dir)
    metadata = load_sample_metadata(clip_dir)
    level = infer_level(metadata.get("position_label"), args.level)
    config = LEVEL_CONFIGS[level]
    height_percentile = float(args.height_percentile or config["height_percentile"])

    rgb_path = find_one(clip_dir, ".rgb.png")
    depth_path = find_one(clip_dir, ".depth_meters.npy")
    intr_path = find_one(clip_dir, ".intrinsics.pkl")
    human_mask_path = clip_dir / "human_mask.npy"
    if not human_mask_path.exists():
        raise FileNotFoundError(f"Human mask not found: {human_mask_path}")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path)
    intrinsics = load_pickle(intr_path)
    human_mask = np.load(human_mask_path).astype(bool)
    rgb = resize_rgb_to_shape(rgb, depth.shape)
    if human_mask.shape != depth.shape:
        raise ValueError("human_mask.npy must have the same shape as depth_meters.npy.")

    valid = valid_depth_mask(depth)
    height, width = depth.shape
    v_frac0, v_frac1 = config["v_fraction"]
    v_range = (int(v_frac0 * height), int(v_frac1 * height))
    human_box = human_bbox(human_mask)
    side = (
        choose_auto_side(valid, human_mask, human_box, v_range=v_range, margin_pixels=args.margin_pixels)
        if args.shelf_side == "auto"
        else args.shelf_side
    )

    rows = np.arange(height)[:, None]
    level_band = (rows >= v_range[0]) & (rows <= v_range[1])
    side_band = side_mask_for_frame(
        width=width,
        human_box=human_box,
        side=side,
        margin_pixels=args.margin_pixels,
    )
    candidate_mask = valid & ~human_mask & level_band & side_band
    candidate_points, candidate_uv = backproject_depth_to_pointcloud(
        depth,
        intrinsics,
        valid_mask=candidate_mask,
        return_uv=True,
    )
    if candidate_points.shape[0] < 100:
        raise ValueError(
            f"Too few shelf/object candidate points ({candidate_points.shape[0]}). "
            "Try --shelf-side auto or override --level."
        )

    floor_y, floor_stats = load_floor_reference(clip_dir, candidate_points)
    heights = floor_y - candidate_points[:, 1]
    plausible = (heights > 0.15) & (heights < 3.2)
    candidate_points = candidate_points[plausible]
    candidate_uv = candidate_uv[plausible]
    heights = heights[plausible]
    if heights.shape[0] < 100:
        raise ValueError("Too few plausible shelf/object height candidates after filtering.")

    filtered_points, open3d_stats = maybe_open3d_filter(candidate_points)
    filtered_heights = floor_y - filtered_points[:, 1]
    target_height_m, height_selection_stats = select_target_height(
        filtered_heights,
        level=level,
        fallback_percentile=height_percentile,
    )
    target_y_m = float(floor_y - target_height_m)
    selected = np.abs(heights - target_height_m) <= 0.06
    if int(selected.sum()) < 50:
        selected = np.abs(heights - target_height_m) <= 0.10
    selected_heights = heights[selected]
    if selected_heights.shape[0] >= 50:
        uncertainty_low, uncertainty_high = np.percentile(selected_heights, [16, 84])
    else:
        uncertainty_low, uncertainty_high = target_height_m - 0.06, target_height_m + 0.06

    human_height_m = load_human_height(clip_dir)
    height_ratio = float(target_height_m / human_height_m) if human_height_m else None
    calibrated_height = (
        float(height_ratio * args.known_human_height_m)
        if height_ratio is not None and args.known_human_height_m is not None
        else None
    )

    preview_path = clip_dir / "shelf_height_preview.png"
    report_path = clip_dir / "shelf_height_report.png"
    stats_path = clip_dir / "shelf_height_estimate.json"
    summary_path = clip_dir / "shelf_height_summary.txt"
    save_height_preview(
        rgb=rgb,
        candidate_uv=candidate_uv,
        selected_uv=candidate_uv[selected],
        v_range=v_range,
        side=side,
        out_path=preview_path,
    )
    save_height_report(
        rgb=rgb,
        candidate_uv=candidate_uv,
        selected_uv=candidate_uv[selected],
        filtered_heights=filtered_heights,
        v_range=v_range,
        side=side,
        target_height_m=target_height_m,
        uncertainty_band_m=(float(uncertainty_low), float(uncertainty_high)),
        height_selection_stats=height_selection_stats,
        human_height_m=human_height_m,
        height_ratio=height_ratio,
        calibrated_height_m=calibrated_height,
        out_path=report_path,
    )

    payload = {
        "status": "shelf_height_estimated",
        "clip_dir": str(clip_dir),
        "sample_role": metadata.get("sample_role"),
        "position_label": metadata.get("position_label"),
        "target_level": level,
        "shelf_side": side,
        "method": "keyframe_depth_roi_level_band_histogram_mode",
        "floor_reference": floor_stats,
        "target_y_m": target_y_m,
        "estimated_shelf_height_m": target_height_m,
        "estimated_shelf_height_uncertainty_band_m": [
            float(uncertainty_low),
            float(uncertainty_high),
        ],
        "height_percentile": height_percentile,
        "height_selection": height_selection_stats,
        "estimated_human_height_m": human_height_m,
        "shelf_to_human_height_ratio": height_ratio,
        "known_human_height_m": args.known_human_height_m,
        "known_human_height_calibrated_shelf_height_m": calibrated_height,
        "candidate_stats": {
            "candidate_count": int(candidate_points.shape[0]),
            "filtered_candidate_count": int(filtered_points.shape[0]),
            "human_bbox_xyxy": list(human_box) if human_box else None,
            "target_v_range_pixels": list(v_range),
            "height_quantiles_m": {
                str(q): float(np.percentile(filtered_heights, q))
                for q in [5, 25, 50, 75, 85, 95]
            },
            "open3d_filter": open3d_stats,
        },
        "inputs": {
            "rgb_png": str(rgb_path),
            "depth_meters_npy": str(depth_path),
            "intrinsics_pkl": str(intr_path),
            "human_mask_npy": str(human_mask_path),
        },
        "outputs": {
            "shelf_height_preview_png": str(preview_path),
            "shelf_height_report_png": str(report_path),
            "shelf_height_summary_txt": str(summary_path),
            "shelf_height_estimate_json": str(stats_path),
        },
        "notes": [
            "Camera Y points down, so height is floor_y_m - target_y_m.",
            "The estimate uses the prepared keyframe: first for lift clips and last for put clips.",
            "This is an automatic ROI estimate; inspect shelf_height_preview.png before treating it as final measurement.",
        ],
    }
    write_json(stats_path, payload)
    write_height_summary(summary_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
