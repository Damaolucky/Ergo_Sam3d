#!/usr/bin/env python3
"""Generate a human mask from RGB and filter depth into a human point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from pipeline_utils import (
    DEFAULT_YOLO_MODEL,
    backproject_depth_to_pointcloud,
    ensure_output_roots,
    load_pickle,
    resolve_in_outputs,
    save_pointcloud_preview,
    write_json,
)


def resolve_clip_dir(clip_arg: str) -> Path:
    """Resolve a clip output directory directly or from the outputs root."""
    return resolve_in_outputs(clip_arg, expect_dir=True, label="Clip directory")


def save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    """Persist a binary mask as a grayscale PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(out_path)


def summarize_mask(mask: np.ndarray, depth_meters: np.ndarray, human_points: np.ndarray) -> dict:
    """Summarize mask coverage and the remaining depth-backed human points."""
    mask_bool = mask.astype(bool)
    valid_depth = np.isfinite(depth_meters) & (depth_meters > 0)
    masked_valid = mask_bool & valid_depth
    stats = {
        "mask_shape": list(mask_bool.shape),
        "mask_num_pixels": int(mask_bool.sum()),
        "mask_ratio": float(mask_bool.mean()),
        "masked_valid_depth_pixels": int(masked_valid.sum()),
        "human_point_count": int(human_points.shape[0]),
    }

    if masked_valid.any():
        values = depth_meters[masked_valid]
        stats.update(
            {
                "human_depth_min_m": float(values.min()),
                "human_depth_max_m": float(values.max()),
                "human_depth_mean_m": float(values.mean()),
                "human_depth_median_m": float(np.median(values)),
            }
        )
    else:
        stats.update(
            {
                "human_depth_min_m": None,
                "human_depth_max_m": None,
                "human_depth_mean_m": None,
                "human_depth_median_m": None,
            }
        )
    return stats


def choose_largest_person_mask(results) -> np.ndarray:
    """Pick the largest YOLO segmentation mask whose class id is `person`."""
    if not results:
        raise RuntimeError("No results returned by YOLO.")

    result = results[0]
    if result.masks is None or result.boxes is None:
        raise RuntimeError("No segmentation masks detected.")

    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks = result.masks.data.detach().cpu().numpy()
    person_indices = [index for index, class_id in enumerate(classes) if class_id == 0]
    if not person_indices:
        raise RuntimeError("No person instance detected.")

    best_idx = None
    best_area = -1
    for index in person_indices:
        area = int((masks[index] > 0.5).sum())
        if area > best_area:
            best_area = area
            best_idx = index

    return (masks[best_idx] > 0.5).astype(np.uint8)


def resize_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to the depth resolution using nearest-neighbor."""
    target_h, target_w = target_shape
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((target_w, target_h), resample=Image.NEAREST)
    return (np.array(image) > 127).astype(np.uint8)


def main() -> None:
    """Parse arguments and save human segmentation outputs for one clip folder."""
    parser = argparse.ArgumentParser(
        description="Generate a human mask and masked human point cloud for one clip directory."
    )
    parser.add_argument(
        "clip_dir",
        help="Clip output folder name or path, e.g. 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4",
    )
    args = parser.parse_args()

    ensure_output_roots()

    clip_dir = resolve_clip_dir(args.clip_dir)
    rgb_path = clip_dir / f"{clip_dir.name}.rgb.png"
    depth_path = clip_dir / f"{clip_dir.name}.depth_meters.npy"
    intr_path = clip_dir / f"{clip_dir.name}.intrinsics.pkl"

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth meters file not found: {depth_path}")
    if not intr_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intr_path}")

    mask_png = clip_dir / "human_mask.png"
    mask_npy = clip_dir / "human_mask.npy"
    masked_depth_npy = clip_dir / "masked_depth_meters.npy"
    human_pc_npy = clip_dir / "human_pointcloud.npy"
    human_pc_preview = clip_dir / "human_pointcloud_preview.png"
    mask_stats_json = clip_dir / "mask_stats.json"

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth_meters = np.load(depth_path)
    intrinsics = load_pickle(intr_path)

    model = YOLO(DEFAULT_YOLO_MODEL)
    results = model.predict(source=rgb, verbose=False, save=False)
    human_mask = choose_largest_person_mask(results).astype(np.uint8)

    print("original mask shape:", human_mask.shape)
    print("depth shape:", depth_meters.shape)
    if human_mask.shape != depth_meters.shape:
        human_mask = resize_mask_to_shape(human_mask, depth_meters.shape)
    print("resized mask shape:", human_mask.shape)

    human_mask_bool = human_mask.astype(bool)
    np.save(mask_npy, human_mask_bool.astype(np.uint8))
    save_mask_png(human_mask_bool, mask_png)

    masked_depth = np.where(human_mask_bool, depth_meters, 0).astype(np.float32)
    np.save(masked_depth_npy, masked_depth)

    human_points = backproject_depth_to_pointcloud(masked_depth, intrinsics)
    np.save(human_pc_npy, human_points)
    save_pointcloud_preview(
        human_points,
        human_pc_preview,
        title="Human Point Cloud Preview (X-Z)",
        empty_message="No valid human 3D points",
    )

    stats = summarize_mask(human_mask_bool, depth_meters, human_points)
    stats.update(
        {
            "clip_dir": str(clip_dir),
            "rgb_path": str(rgb_path),
            "depth_meters_path": str(depth_path),
            "intrinsics_path": str(intr_path),
            "outputs": {
                "human_mask_png": str(mask_png),
                "human_mask_npy": str(mask_npy),
                "masked_depth_meters_npy": str(masked_depth_npy),
                "human_pointcloud_npy": str(human_pc_npy),
                "human_pointcloud_preview_png": str(human_pc_preview),
            },
        }
    )
    write_json(mask_stats_json, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
