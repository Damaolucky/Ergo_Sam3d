#!/usr/bin/env python3
"""Estimate a hand-anchored target height from the prepared keyframe sample.

The prepared keyframe is action-aware: `*_lift.mp4` clips use the first frame at
the source shelf/object position, while `*_put.mp4` clips use the last frame at
the destination position. The current height estimate is intentionally tied to
the operator's final hand position in that keyframe, because the hand is the
most direct cue for where the object is being picked from or placed to.

The depth camera frame uses `Y` pointing downward, so a vertical height above
the floor is computed as:

    height_m = floor_y_m - target_y_m

The floor reference is taken from the aligned human mesh feet when available.
The primary method detects the target-side hand silhouette extremity in the
human mask, samples a local depth patch just inside that extremity, and reports
the resulting hand height. A simpler shelf-side ROI estimate remains as a
fallback when the local hand depth is missing.
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


LEVEL_CONFIGS: dict[str, dict[str, Any]] = {
    "high": {
        "bbox_hand_v_fraction": (0.15, 0.65),
        "roi_v_fraction": (0.10, 0.36),
        "fallback_percentile": 85.0,
    },
    "mid": {
        "bbox_hand_v_fraction": (0.20, 0.82),
        "roi_v_fraction": (0.30, 0.62),
        "fallback_percentile": 55.0,
    },
    "low": {
        "bbox_hand_v_fraction": (0.35, 0.98),
        "roi_v_fraction": (0.52, 0.92),
        "fallback_percentile": 25.0,
    },
}


PREVIEW_RNG = np.random.default_rng(0)


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
    """Infer the target level from metadata unless the caller overrides it."""
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
    """Build a simple left/right search band outside the human bbox."""
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
    roi_v_range: tuple[int, int],
    margin_pixels: int,
) -> str:
    """Choose the side with stronger non-human depth support near the target band."""
    height, width = valid.shape
    rows = np.arange(height)[:, None]
    level_band = (rows >= roi_v_range[0]) & (rows <= roi_v_range[1])
    scores: dict[str, int] = {}
    for side in ("left", "right"):
        side_mask = side_mask_for_frame(
            width=width,
            human_box=human_box,
            side=side,
            margin_pixels=margin_pixels,
        )
        scores[side] = int((valid & ~human_mask & level_band & side_mask).sum())
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


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert a boolean 1D mask into inclusive index runs."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def detect_hand_anchor(
    human_mask: np.ndarray,
    *,
    side: str,
    level: str,
    human_box: tuple[int, int, int, int] | None,
) -> dict[str, Any] | None:
    """Detect the target-side hand anchor from the mask silhouette extremity."""
    if human_box is None:
        return None

    xmin, ymin, xmax, ymax = human_box
    bbox_height = ymax - ymin + 1
    bbox_width = xmax - xmin + 1
    if bbox_height < 20 or bbox_width < 20:
        return None

    frac0, frac1 = LEVEL_CONFIGS[level]["bbox_hand_v_fraction"]
    row_start = max(ymin, ymin + int(frac0 * bbox_height))
    row_end = min(ymax, ymin + int(frac1 * bbox_height))

    rows: list[int] = []
    edges: list[int] = []
    for row_index in range(row_start, row_end + 1):
        cols = np.where(human_mask[row_index])[0]
        if cols.size == 0:
            continue
        rows.append(row_index)
        edges.append(int(cols.min()) if side == "left" else int(cols.max()))

    if len(rows) < 8:
        return None

    rows_arr = np.asarray(rows, dtype=np.int32)
    edges_arr = np.asarray(edges, dtype=np.int32)
    best_edge = int(edges_arr.min()) if side == "left" else int(edges_arr.max())
    edge_tolerance = max(10, int(0.05 * bbox_width))
    if side == "left":
        near_best = edges_arr <= best_edge + edge_tolerance
    else:
        near_best = edges_arr >= best_edge - edge_tolerance

    runs = contiguous_true_runs(near_best)
    if not runs:
        best_local_index = int(np.argmin(edges_arr) if side == "left" else np.argmax(edges_arr))
        anchor_row = int(rows_arr[best_local_index])
        anchor_edge = int(edges_arr[best_local_index])
        run_rows = np.asarray([anchor_row], dtype=np.int32)
    else:
        best_run = max(runs, key=lambda run: (run[1] - run[0] + 1, float(rows_arr[run[0] : run[1] + 1].mean())))
        run_rows = rows_arr[best_run[0] : best_run[1] + 1]
        anchor_row = int(np.median(run_rows))
        row_offset = int(np.argmin(np.abs(rows_arr - anchor_row)))
        anchor_edge = int(edges_arr[row_offset])

    return {
        "method": "target_side_mask_extremity",
        "side": side,
        "level": level,
        "search_rows_pixels": [int(row_start), int(row_end)],
        "anchor_uv_px": [int(anchor_edge), int(anchor_row)],
        "target_side_edge_best_px": best_edge,
        "target_side_edge_tolerance_px": int(edge_tolerance),
        "anchor_row_cluster_pixels": [int(run_rows.min()), int(run_rows.max())],
        "human_bbox_xyxy": [int(xmin), int(ymin), int(xmax), int(ymax)],
    }


def collect_hand_patch(
    depth: np.ndarray,
    intrinsics: Any,
    human_mask: np.ndarray,
    *,
    side: str,
    anchor_uv: tuple[int, int],
    human_box: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect a local depth patch just inside the detected hand extremity."""
    if human_box is None:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
            {"status": "missing_human_box"},
        )

    xmin, ymin, xmax, ymax = human_box
    anchor_x, anchor_y = anchor_uv
    bbox_height = ymax - ymin + 1
    bbox_width = xmax - xmin + 1
    row_half_span = max(8, int(0.03 * bbox_height))
    inward_span = max(18, int(0.10 * bbox_width))

    attempts = [
        {"row_half_span": row_half_span, "inward_span": inward_span},
        {"row_half_span": row_half_span * 2, "inward_span": int(inward_span * 1.4)},
        {"row_half_span": row_half_span * 3, "inward_span": int(inward_span * 1.8)},
    ]

    for attempt in attempts:
        y0 = max(ymin, anchor_y - attempt["row_half_span"])
        y1 = min(ymax, anchor_y + attempt["row_half_span"])
        if side == "right":
            x0 = max(xmin, anchor_x - attempt["inward_span"])
            x1 = min(xmax, anchor_x)
        else:
            x0 = max(xmin, anchor_x)
            x1 = min(xmax, anchor_x + attempt["inward_span"])

        local_mask = np.zeros_like(human_mask, dtype=bool)
        local_mask[y0 : y1 + 1, x0 : x1 + 1] = True
        valid = human_mask & local_mask & valid_depth_mask(depth)
        points, uv = backproject_depth_to_pointcloud(depth, intrinsics, valid_mask=valid, return_uv=True)
        if points.shape[0] >= 20:
            return points.astype(np.float32), uv.astype(np.int32), {
                "status": "ok",
                "patch_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "row_half_span_px": int(attempt["row_half_span"]),
                "inward_span_px": int(attempt["inward_span"]),
                "point_count": int(points.shape[0]),
            }

    return (
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 2), dtype=np.int32),
        {
            "status": "too_few_valid_depth_points",
            "anchor_uv_px": [int(anchor_x), int(anchor_y)],
            "attempts": attempts,
        },
    )


def fallback_roi_height(
    *,
    depth: np.ndarray,
    intrinsics: Any,
    human_mask: np.ndarray,
    human_box: tuple[int, int, int, int] | None,
    level: str,
    side: str,
    floor_y: float,
    margin_pixels: int,
) -> tuple[float, np.ndarray, np.ndarray, tuple[float, float], dict[str, Any]]:
    """Estimate a target height from a simple shelf-side depth ROI."""
    valid = valid_depth_mask(depth)
    height, width = depth.shape
    v_frac0, v_frac1 = LEVEL_CONFIGS[level]["roi_v_fraction"]
    v_range = (int(v_frac0 * height), int(v_frac1 * height))
    rows = np.arange(height)[:, None]
    level_band = (rows >= v_range[0]) & (rows <= v_range[1])
    side_band = side_mask_for_frame(
        width=width,
        human_box=human_box,
        side=side,
        margin_pixels=margin_pixels,
    )
    candidate_mask = valid & ~human_mask & level_band & side_band
    candidate_points, candidate_uv = backproject_depth_to_pointcloud(
        depth,
        intrinsics,
        valid_mask=candidate_mask,
        return_uv=True,
    )
    if candidate_points.shape[0] < 20:
        raise ValueError(
            "Too few fallback shelf/object candidate points. "
            "Try --shelf-side auto or check the human mask quality."
        )

    heights = floor_y - candidate_points[:, 1]
    plausible = (heights > 0.15) & (heights < 3.2)
    candidate_points = candidate_points[plausible]
    candidate_uv = candidate_uv[plausible]
    heights = heights[plausible]
    if heights.shape[0] < 20:
        raise ValueError("Too few plausible fallback shelf/object height candidates after filtering.")

    percentile = float(LEVEL_CONFIGS[level]["fallback_percentile"])
    target_height_m = float(np.percentile(heights, percentile))
    selected = np.abs(heights - target_height_m) <= 0.06
    if int(selected.sum()) < 10:
        selected = np.abs(heights - target_height_m) <= 0.10
    selected_heights = heights[selected]
    if selected_heights.shape[0] >= 10:
        uncertainty_low, uncertainty_high = np.percentile(selected_heights, [16, 84])
    else:
        uncertainty_low, uncertainty_high = target_height_m - 0.06, target_height_m + 0.06

    stats = {
        "method": "shelf_side_depth_roi_percentile_fallback",
        "target_v_range_pixels": [int(v_range[0]), int(v_range[1])],
        "candidate_count": int(candidate_points.shape[0]),
        "percentile": percentile,
        "patch_quantiles_m": {str(q): float(np.percentile(heights, q)) for q in [5, 25, 50, 75, 95]},
    }
    return (
        target_height_m,
        candidate_points.astype(np.float32),
        candidate_uv[selected].astype(np.int32),
        (float(uncertainty_low), float(uncertainty_high)),
        stats,
    )


def save_height_preview(
    *,
    rgb: np.ndarray,
    human_box: tuple[int, int, int, int] | None,
    side: str,
    hand_search_rows: tuple[int, int] | None,
    hand_anchor_uv: tuple[int, int] | None,
    local_patch_xyxy: tuple[int, int, int, int] | None,
    selected_uv: np.ndarray,
    out_path: Path,
) -> None:
    """Save an RGB overlay showing the hand anchor and local height patch."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(rgb)

    if human_box is not None:
        xmin, ymin, xmax, ymax = human_box
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="#7aa5c7",
                linewidth=1.2,
                linestyle="--",
                label="human bbox",
            )
        )

    if hand_search_rows is not None:
        ax.axhspan(
            hand_search_rows[0],
            hand_search_rows[1],
            color="orange",
            alpha=0.10,
            label=f"{side} hand search band",
        )

    if local_patch_xyxy is not None:
        x0, y0, x1, y1 = local_patch_xyxy
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="red",
                linewidth=1.4,
                label="local hand patch",
            )
        )

    if selected_uv.size:
        sample = selected_uv
        if sample.shape[0] > 2500:
            idx = PREVIEW_RNG.choice(sample.shape[0], 2500, replace=False)
            sample = sample[idx]
        ax.scatter(sample[:, 0], sample[:, 1], s=3, c="red", alpha=0.35, label="selected depth pixels")

    if hand_anchor_uv is not None:
        ax.scatter(
            [hand_anchor_uv[0]],
            [hand_anchor_uv[1]],
            s=70,
            c="yellow",
            edgecolors="black",
            linewidths=0.8,
            marker="*",
            label="hand anchor",
        )

    ax.set_title("Hand-anchored target height preview")
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
    human_box: tuple[int, int, int, int] | None,
    side: str,
    hand_search_rows: tuple[int, int] | None,
    hand_anchor_uv: tuple[int, int] | None,
    local_patch_xyxy: tuple[int, int, int, int] | None,
    selected_uv: np.ndarray,
    heights: np.ndarray,
    target_height_m: float,
    uncertainty_band_m: tuple[float, float],
    method: str,
    human_height_m: float | None,
    height_ratio: float | None,
    calibrated_height_m: float | None,
    out_path: Path,
) -> None:
    """Save a combined visual report for manual height inspection."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(rgb)
    if human_box is not None:
        xmin, ymin, xmax, ymax = human_box
        axes[0].add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="#7aa5c7",
                linewidth=1.2,
                linestyle="--",
                label="human bbox",
            )
        )
    if hand_search_rows is not None:
        axes[0].axhspan(
            hand_search_rows[0],
            hand_search_rows[1],
            color="orange",
            alpha=0.10,
            label=f"{side} hand search band",
        )
    if local_patch_xyxy is not None:
        x0, y0, x1, y1 = local_patch_xyxy
        axes[0].add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="red",
                linewidth=1.4,
                label="local hand patch",
            )
        )
    if selected_uv.size:
        sample = selected_uv
        if sample.shape[0] > 2500:
            idx = PREVIEW_RNG.choice(sample.shape[0], 2500, replace=False)
            sample = sample[idx]
        axes[0].scatter(sample[:, 0], sample[:, 1], s=3, c="red", alpha=0.35, label="selected depth pixels")
    if hand_anchor_uv is not None:
        axes[0].scatter(
            [hand_anchor_uv[0]],
            [hand_anchor_uv[1]],
            s=70,
            c="yellow",
            edgecolors="black",
            linewidths=0.8,
            marker="*",
            label="hand anchor",
        )
    axes[0].set_title("Keyframe hand anchor")
    axes[0].set_xlim(0, rgb.shape[1])
    axes[0].set_ylim(rgb.shape[0], 0)
    axes[0].axis("off")
    axes[0].legend(loc="lower left")

    axes[1].hist(heights, bins=min(60, max(12, heights.shape[0] // 3)), color="#8fa7b3", alpha=0.82)
    axes[1].axvspan(
        uncertainty_band_m[0],
        uncertainty_band_m[1],
        color="red",
        alpha=0.16,
        label="uncertainty band",
    )
    axes[1].axvline(target_height_m, color="red", linewidth=2.0, label="selected height")
    axes[1].set_xlabel("height above floor (meters)")
    axes[1].set_ylabel("point count")
    axes[1].set_title("Local depth-patch height distribution")
    axes[1].legend(loc="upper right")

    lines = [
        f"estimated target height: {target_height_m:.3f} m",
        f"uncertainty band: [{uncertainty_band_m[0]:.3f}, {uncertainty_band_m[1]:.3f}] m",
        f"method: {method}",
    ]
    if human_height_m is not None:
        lines.append(f"estimated human height: {human_height_m:.3f} m")
    if height_ratio is not None:
        lines.append(f"target / human ratio: {height_ratio:.3f}")
    if calibrated_height_m is not None:
        lines.append(f"known-height calibrated target height: {calibrated_height_m:.3f} m")
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
    """Write a concise text summary of the target-height estimate."""
    calibrated = payload.get("known_human_height_calibrated_target_height_m")
    lines = [
        "Hand-anchored target height estimate",
        f"clip_dir: {payload['clip_dir']}",
        f"position_label: {payload.get('position_label')}",
        f"target_level: {payload.get('target_level')}",
        f"shelf_side: {payload.get('shelf_side')}",
        f"estimated_target_height_m: {payload['estimated_target_height_m']:.4f}",
        (
            "estimated_target_height_uncertainty_band_m: "
            f"[{payload['estimated_target_height_uncertainty_band_m'][0]:.4f}, "
            f"{payload['estimated_target_height_uncertainty_band_m'][1]:.4f}]"
        ),
        f"estimated_human_height_m: {payload.get('estimated_human_height_m')}",
        f"target_to_human_height_ratio: {payload.get('target_to_human_height_ratio')}",
        f"known_human_height_calibrated_target_height_m: {calibrated}",
        f"method: {payload.get('method')}",
        "view: shelf_height_report.png and shelf_height_preview.png",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Estimate the hand-anchored target height for a prepared keyframe sample."""
    parser = argparse.ArgumentParser(
        description="Estimate the target height from the final hand position in a prepared keyframe folder."
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
        help="Which image side contains the target shelf/object region. Default: right.",
    )
    parser.add_argument(
        "--known-human-height-m",
        type=float,
        default=None,
        help="Optional real human height. If provided, also report the calibrated target height.",
    )
    parser.add_argument(
        "--margin-pixels",
        type=int,
        default=24,
        help="Horizontal margin away from the human bbox for the fallback shelf-side ROI.",
    )
    args = parser.parse_args()

    ensure_output_roots()
    clip_dir = resolve_clip_dir(args.clip_dir)
    metadata = load_sample_metadata(clip_dir)
    level = infer_level(metadata.get("position_label"), args.level)

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
    human_box = human_bbox(human_mask)
    if human_box is None:
        raise ValueError("Human mask is empty, so hand height cannot be estimated.")

    roi_v_frac0, roi_v_frac1 = LEVEL_CONFIGS[level]["roi_v_fraction"]
    roi_v_range = (int(roi_v_frac0 * depth.shape[0]), int(roi_v_frac1 * depth.shape[0]))
    side = (
        choose_auto_side(valid, human_mask, human_box, roi_v_range=roi_v_range, margin_pixels=args.margin_pixels)
        if args.shelf_side == "auto"
        else args.shelf_side
    )

    fallback_floor_points = backproject_depth_to_pointcloud(depth, intrinsics, valid_mask=valid, return_uv=False)
    floor_y, floor_stats = load_floor_reference(clip_dir, fallback_floor_points)

    hand_anchor = detect_hand_anchor(human_mask, side=side, level=level, human_box=human_box)
    method = "hand_final_height_local_depth_patch"
    target_height_m: float
    uncertainty_band_m: tuple[float, float]
    selected_uv: np.ndarray
    hand_patch_stats: dict[str, Any]
    patch_heights: np.ndarray
    preview_patch_xyxy: tuple[int, int, int, int] | None = None

    if hand_anchor is not None:
        anchor_uv = tuple(int(v) for v in hand_anchor["anchor_uv_px"])
        hand_points, hand_uv, hand_patch_stats = collect_hand_patch(
            depth,
            intrinsics,
            human_mask,
            side=side,
            anchor_uv=anchor_uv,
            human_box=human_box,
        )
        if hand_patch_stats.get("status") == "ok":
            preview_patch_xyxy = tuple(int(v) for v in hand_patch_stats["patch_xyxy"])
            patch_heights = floor_y - hand_points[:, 1]
            plausible = (patch_heights > 0.15) & (patch_heights < 3.2)
            hand_points = hand_points[plausible]
            hand_uv = hand_uv[plausible]
            patch_heights = patch_heights[plausible]
        else:
            patch_heights = np.empty((0,), dtype=np.float32)
    else:
        hand_patch_stats = {"status": "hand_anchor_not_found"}
        hand_uv = np.empty((0, 2), dtype=np.int32)
        patch_heights = np.empty((0,), dtype=np.float32)

    if patch_heights.shape[0] >= 20:
        target_height_m = float(np.median(patch_heights))
        uncertainty_low, uncertainty_high = np.percentile(patch_heights, [16, 84])
        selected = np.abs(patch_heights - target_height_m) <= 0.05
        if int(selected.sum()) < 10:
            selected = np.abs(patch_heights - target_height_m) <= 0.08
        selected_uv = hand_uv[selected]
        uncertainty_band_m = (float(uncertainty_low), float(uncertainty_high))
        target_selection_stats = {
            "method": "local_hand_patch_median",
            "point_count": int(patch_heights.shape[0]),
            "patch_quantiles_m": {str(q): float(np.percentile(patch_heights, q)) for q in [5, 25, 50, 75, 95]},
        }
    else:
        method = "shelf_side_depth_roi_percentile_fallback"
        hand_anchor = hand_anchor or {
            "method": "unavailable",
            "side": side,
            "level": level,
        }
        target_height_m, _fallback_points, selected_uv, uncertainty_band_m, target_selection_stats = fallback_roi_height(
            depth=depth,
            intrinsics=intrinsics,
            human_mask=human_mask,
            human_box=human_box,
            level=level,
            side=side,
            floor_y=floor_y,
            margin_pixels=args.margin_pixels,
        )
        patch_heights = np.asarray([target_height_m], dtype=np.float32)

    target_y_m = float(floor_y - target_height_m)
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

    hand_search_rows = None
    hand_anchor_uv = None
    if hand_anchor is not None:
        rows = hand_anchor.get("search_rows_pixels")
        if rows is not None:
            hand_search_rows = (int(rows[0]), int(rows[1]))
        anchor = hand_anchor.get("anchor_uv_px")
        if anchor is not None:
            hand_anchor_uv = (int(anchor[0]), int(anchor[1]))

    save_height_preview(
        rgb=rgb,
        human_box=human_box,
        side=side,
        hand_search_rows=hand_search_rows,
        hand_anchor_uv=hand_anchor_uv,
        local_patch_xyxy=preview_patch_xyxy,
        selected_uv=selected_uv,
        out_path=preview_path,
    )
    save_height_report(
        rgb=rgb,
        human_box=human_box,
        side=side,
        hand_search_rows=hand_search_rows,
        hand_anchor_uv=hand_anchor_uv,
        local_patch_xyxy=preview_patch_xyxy,
        selected_uv=selected_uv,
        heights=patch_heights,
        target_height_m=target_height_m,
        uncertainty_band_m=uncertainty_band_m,
        method=method,
        human_height_m=human_height_m,
        height_ratio=height_ratio,
        calibrated_height_m=calibrated_height,
        out_path=report_path,
    )

    payload = {
        "status": "hand_anchored_target_height_estimated",
        "clip_dir": str(clip_dir),
        "sample_role": metadata.get("sample_role"),
        "position_label": metadata.get("position_label"),
        "target_level": level,
        "shelf_side": side,
        "method": method,
        "floor_reference": floor_stats,
        "target_y_m": target_y_m,
        "estimated_target_height_m": target_height_m,
        "estimated_target_height_uncertainty_band_m": [
            float(uncertainty_band_m[0]),
            float(uncertainty_band_m[1]),
        ],
        "estimated_hand_height_m": target_height_m if method == "hand_final_height_local_depth_patch" else None,
        "estimated_shelf_height_m": target_height_m,
        "estimated_shelf_height_uncertainty_band_m": [
            float(uncertainty_band_m[0]),
            float(uncertainty_band_m[1]),
        ],
        "estimated_human_height_m": human_height_m,
        "target_to_human_height_ratio": height_ratio,
        "shelf_to_human_height_ratio": height_ratio,
        "known_human_height_m": args.known_human_height_m,
        "known_human_height_calibrated_target_height_m": calibrated_height,
        "known_human_height_calibrated_shelf_height_m": calibrated_height,
        "hand_anchor": hand_anchor,
        "hand_patch": hand_patch_stats,
        "target_selection": target_selection_stats,
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
            "The primary estimate is hand-anchored: it measures the final hand height in the selected keyframe, not a full cabinet surface reconstruction.",
            "Because no color intrinsics / color-depth extrinsics are available, the RGB human mask is resized to depth resolution and remains approximate.",
            "Inspect shelf_height_preview.png and shelf_height_report.png before treating the result as a final physical measurement.",
        ],
    }
    write_json(stats_path, payload)
    write_height_summary(summary_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
