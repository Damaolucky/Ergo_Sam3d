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
    load_json,
    load_pickle,
    pointcloud_colors_from_rgb,
    resolve_in_outputs,
    resize_rgb_to_shape,
    save_pointcloud_preview,
    save_pointcloud_ply,
    valid_depth_mask,
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


def _apply_inverse_brown_conrady_distortion(
    x: np.ndarray,
    y: np.ndarray,
    coeffs: list[float],
    iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply inverse Brown-Conrady distortion to normalized image coordinates.

    Matches the RealSense rs2_project_point_to_pixel implementation for the
    RS2_DISTORTION_INVERSE_BROWN_CONRADY model: iteratively solve for the
    distorted coordinates from the undistorted ones.
    """
    k1, k2, p1, p2, k3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]
    xd, yd = x.copy(), y.copy()
    for _ in range(iterations):
        r2 = xd * xd + yd * yd
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        tangential_x = 2.0 * p1 * xd * yd + p2 * (r2 + 2.0 * xd * xd)
        tangential_y = 2.0 * p2 * xd * yd + p1 * (r2 + 2.0 * yd * yd)
        xd = x * radial + tangential_x
        yd = y * radial + tangential_y
    return xd, yd


def _project_depth_pixels_to_color(
    depth_meters: np.ndarray,
    calibration: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project every depth pixel into color image coordinates.

    Returns (u_color, v_color, valid) arrays at depth resolution.
    """
    depth_h, depth_w = depth_meters.shape

    di = calibration["depth_intrinsics"]
    d_fx, d_fy = float(di["fx"]), float(di["fy"])
    d_cx, d_cy = float(di["ppx"]), float(di["ppy"])

    ci = calibration["color_intrinsics"]
    c_fx, c_fy = float(ci["fx"]), float(ci["fy"])
    c_cx, c_cy = float(ci["ppx"]), float(ci["ppy"])
    distortion_model = ci.get("model", "")
    coeffs = ci.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])

    ext = calibration["depth_to_color_extrinsics"]
    R = np.array(ext["rotation_row_major_3x3"], dtype=np.float64).reshape(3, 3)
    t = np.array(ext["translation_meters_xyz"], dtype=np.float64).reshape(3, 1)

    u_d, v_d = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    z = depth_meters.astype(np.float64)
    valid = np.isfinite(z) & (z > 0)

    x_3d = (u_d - d_cx) * z / d_fx
    y_3d = (v_d - d_cy) * z / d_fy

    pts_flat = np.stack([x_3d.ravel(), y_3d.ravel(), z.ravel()], axis=0)
    pts_color = R @ pts_flat + t

    xn = pts_color[0] / pts_color[2]
    yn = pts_color[1] / pts_color[2]

    if "inverse_brown_conrady" in distortion_model and any(c != 0 for c in coeffs):
        xn, yn = _apply_inverse_brown_conrady_distortion(xn, yn, coeffs)

    u_c = (c_fx * xn + c_cx).reshape(depth_h, depth_w)
    v_c = (c_fy * yn + c_cy).reshape(depth_h, depth_w)

    return u_c, v_c, valid


def sample_color_for_depth(
    rgb: np.ndarray,
    depth_meters: np.ndarray,
    calibration: dict,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Sample RGB colors for each depth pixel by projecting into color space.

    Returns an (H, W, 3) uint8 image at depth resolution with calibrated
    color alignment.
    """
    depth_h, depth_w = depth_meters.shape
    color_h, color_w = rgb.shape[:2]

    u_c, v_c, depth_valid = _project_depth_pixels_to_color(depth_meters, calibration)
    u_c_int = np.rint(u_c).astype(np.int32)
    v_c_int = np.rint(v_c).astype(np.int32)

    if valid_mask is None:
        valid_mask = depth_valid

    in_bounds = (
        valid_mask
        & (u_c_int >= 0) & (u_c_int < color_w)
        & (v_c_int >= 0) & (v_c_int < color_h)
    )

    aligned_rgb = np.zeros((depth_h, depth_w, 3), dtype=np.uint8)
    aligned_rgb[in_bounds] = rgb[v_c_int[in_bounds], u_c_int[in_bounds]]
    return aligned_rgb


def reproject_color_mask_to_depth(
    color_mask: np.ndarray,
    depth_meters: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    """Reproject a binary mask from color space into depth space using calibration."""
    depth_h, depth_w = depth_meters.shape
    color_h, color_w = color_mask.shape[:2]

    u_c, v_c, valid = _project_depth_pixels_to_color(depth_meters, calibration)
    u_c_int = np.rint(u_c).astype(np.int32)
    v_c_int = np.rint(v_c).astype(np.int32)

    in_bounds = (
        valid
        & (u_c_int >= 0) & (u_c_int < color_w)
        & (v_c_int >= 0) & (v_c_int < color_h)
    )

    depth_mask = np.zeros((depth_h, depth_w), dtype=np.uint8)
    depth_mask[in_bounds] = color_mask[v_c_int[in_bounds], u_c_int[in_bounds]]

    return depth_mask


def main() -> None:
    """Parse arguments and save human segmentation outputs for one clip folder."""
    parser = argparse.ArgumentParser(
        description="Generate a human mask and masked human point cloud for one clip directory."
    )
    parser.add_argument(
        "clip_dir",
        help="Clip output folder name or path, e.g. 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4",
    )
    parser.add_argument(
        "--calibration-json",
        default=None,
        help="Path to camera calibration JSON with color_intrinsics, depth_intrinsics, "
             "and depth_to_color_extrinsics. When provided, the YOLO mask is geometrically "
             "reprojected from color to depth space instead of naively resized.",
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

    calibration = None
    if args.calibration_json is not None:
        calib_path = Path(args.calibration_json)
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration JSON not found: {calib_path}")
        calibration = load_json(calib_path)
        print(f"loaded calibration: serial={calibration.get('serial_number', 'unknown')}")

    model = YOLO(DEFAULT_YOLO_MODEL)
    results = model.predict(source=rgb, verbose=False, save=False)
    color_mask = choose_largest_person_mask(results).astype(np.uint8)

    print("YOLO mask shape:", color_mask.shape)
    print("RGB shape:", rgb.shape[:2])
    print("depth shape:", depth_meters.shape)

    # YOLO may produce masks at a lower resolution than the input RGB image.
    # Resize to match the actual color image resolution before reprojection,
    # since the color intrinsics correspond to the full RGB resolution.
    if color_mask.shape != rgb.shape[:2]:
        color_mask = resize_mask_to_shape(color_mask, rgb.shape[:2])
        print("color mask resized to RGB resolution:", color_mask.shape)

    if calibration is not None:
        print("reprojecting mask from color to depth using calibration extrinsics")
        human_mask = reproject_color_mask_to_depth(color_mask, depth_meters, calibration)
    elif color_mask.shape != depth_meters.shape:
        print("WARNING: no calibration provided, falling back to naive resize")
        human_mask = resize_mask_to_shape(color_mask, depth_meters.shape)
    else:
        human_mask = color_mask
    print("depth-space mask shape:", human_mask.shape)

    human_mask_bool = human_mask.astype(bool)
    np.save(mask_npy, human_mask_bool.astype(np.uint8))
    save_mask_png(human_mask_bool, mask_png)

    base_valid_mask = valid_depth_mask(depth_meters)
    masked_valid = human_mask_bool & base_valid_mask
    masked_depth = np.where(masked_valid, depth_meters, 0).astype(np.float32)
    np.save(masked_depth_npy, masked_depth)

    human_points, human_uv = backproject_depth_to_pointcloud(
        depth_meters,
        intrinsics,
        valid_mask=masked_valid,
        return_uv=True,
    )
    if calibration is not None:
        aligned_rgb = sample_color_for_depth(rgb, depth_meters, calibration, valid_mask=masked_valid)
        human_colors = aligned_rgb[masked_valid]
    else:
        rgb_resized = resize_rgb_to_shape(rgb, depth_meters.shape)
        human_colors = pointcloud_colors_from_rgb(rgb_resized, human_uv)
    np.save(human_pc_npy, human_points)
    human_pc_ply = clip_dir / "human_pointcloud_rgb.ply"
    save_pointcloud_ply(human_pc_ply, human_points, colors=human_colors)
    save_pointcloud_preview(
        human_points,
        human_pc_preview,
        title="Human Point Cloud Preview",
        empty_message="No valid human 3D points",
        colors=human_colors,
    )

    stats = summarize_mask(human_mask_bool, masked_depth, human_points)
    mask_method = "calibrated_reprojection" if calibration is not None else "naive_resize"
    stats.update(
        {
            "clip_dir": str(clip_dir),
            "rgb_path": str(rgb_path),
            "depth_meters_path": str(depth_path),
            "intrinsics_path": str(intr_path),
            "mask_alignment_method": mask_method,
            "calibration_json": str(args.calibration_json) if args.calibration_json else None,
            "calibration_serial": calibration.get("serial_number") if calibration else None,
            "outputs": {
                "human_mask_png": str(mask_png),
                "human_mask_npy": str(mask_npy),
                "masked_depth_meters_npy": str(masked_depth_npy),
                "human_pointcloud_npy": str(human_pc_npy),
                "human_pointcloud_rgb_ply": str(human_pc_ply),
                "human_pointcloud_preview_png": str(human_pc_preview),
            },
        }
    )
    write_json(mask_stats_json, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
