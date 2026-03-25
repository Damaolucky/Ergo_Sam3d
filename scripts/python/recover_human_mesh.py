#!/usr/bin/env python3
"""Recover a human mesh for one clip using 4D-Humans/HMR2 and the existing mask."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline_utils import ensure_output_roots, resolve_in_outputs, write_json


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def resolve_clip_dir(clip_arg: str) -> Path:
    """Resolve a clip output directory directly or from the outputs root."""
    return resolve_in_outputs(clip_arg, expect_dir=True, label="Clip directory")


def maybe_add_hmr2_repo(repo_path: Path | None) -> Path | None:
    """Optionally add a 4D-Humans repo checkout to `sys.path` before import."""
    if repo_path is None:
        return None

    resolved = repo_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"HMR2 repo not found: {resolved}")

    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return resolved


def load_hmr2_modules(repo_path: Path | None) -> dict[str, Any]:
    """Import the HMR2 stack lazily so this script can exist without the dependency installed."""
    maybe_add_hmr2_repo(repo_path)
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ.get("ERGO_OPENGL_PLATFORM", "egl"))

    try:
        import cv2
        import torch
        from hmr2.configs import CACHE_DIR_4DHUMANS
        from hmr2.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD, ViTDetDataset
        from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
        from hmr2.utils import recursive_to
    except Exception as exc:
        raise RuntimeError(
            "HMR2 dependencies are not available. Run scripts/bash/setup_hmr2.sh "
            "or set ERGO_HMR2_PYTHON / ERGO_HMR2_REPO before running this stage."
        ) from exc

    return {
        "cv2": cv2,
        "torch": torch,
        "CACHE_DIR_4DHUMANS": CACHE_DIR_4DHUMANS,
        "DEFAULT_CHECKPOINT": DEFAULT_CHECKPOINT,
        "DEFAULT_MEAN": DEFAULT_MEAN,
        "DEFAULT_STD": DEFAULT_STD,
        "ViTDetDataset": ViTDetDataset,
        "download_models": download_models,
        "load_hmr2": load_hmr2,
        "recursive_to": recursive_to,
    }


def load_mask_bbox(mask_path: Path, image_shape: tuple[int, int], padding_scale: float) -> np.ndarray:
    """Compute a padded XYXY bounding box from the existing human mask."""
    mask = np.asarray(np.load(mask_path)).astype(bool)
    if mask.shape != image_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match RGB shape {image_shape}. "
            "Run the verified human-mask stage first."
        )

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError(f"Mask contains no foreground pixels: {mask_path}")

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * padding_scale

    height, width = image_shape
    side = max(side, 32.0)

    x1 = max(0.0, cx - side / 2.0)
    y1 = max(0.0, cy - side / 2.0)
    x2 = min(float(width), cx + side / 2.0)
    y2 = min(float(height), cy + side / 2.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a simple OBJ mesh with 1-indexed triangular faces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def save_mesh_preview(
    rgb_image: np.ndarray,
    vertices_camera: np.ndarray,
    joints_camera: np.ndarray,
    bbox_xyxy: np.ndarray,
    out_png: Path,
) -> None:
    """Save a lightweight preview that combines RGB, mesh X-Z, and joints X-Z."""
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB Sample")
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    axes[0].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], linewidth=2)
    axes[0].axis("off")

    axes[1].scatter(vertices_camera[:, 0], vertices_camera[:, 2], s=0.1)
    axes[1].set_title("Recovered Mesh Vertices (X-Z)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")

    axes[2].scatter(vertices_camera[:, 0], vertices_camera[:, 2], s=0.05, alpha=0.15)
    axes[2].scatter(joints_camera[:, 0], joints_camera[:, 2], s=12)
    axes[2].set_title("Recovered Joints (X-Z)")
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Z (m)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def tensor_to_list(value: Any) -> Any:
    """Convert numpy/torch-like objects to JSON-serializable lists."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_hmr2_checkpoint_compat(load_hmr2_fn: Any, checkpoint_path: str, torch_module: Any) -> tuple[Any, Any]:
    """Load an HMR2 checkpoint with PyTorch >= 2.6 compatibility.

    PyTorch 2.6 changed `torch.load(..., weights_only=...)` to default to `True`,
    while the current upstream 4D-Humans loader still expects the historical
    `False` behavior for its Lightning checkpoint. The checkpoint in this stage
    is the official HMR2 release artifact downloaded by upstream code, so we
    explicitly restore the legacy behavior only for this trusted load call.
    """
    original_torch_load = torch_module.load

    def compat_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch_module.load = compat_torch_load
    try:
        return load_hmr2_fn(checkpoint_path)
    finally:
        torch_module.load = original_torch_load


def main() -> None:
    """Run HMR2 mesh recovery for one clip directory and save mesh artifacts."""
    parser = argparse.ArgumentParser(
        description="Recover a human mesh for one clip using 4D-Humans/HMR2 and the existing human mask."
    )
    parser.add_argument(
        "clip_dir",
        help="Clip output folder name or path, e.g. 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4",
    )
    parser.add_argument(
        "--hmr2-repo",
        default=os.environ.get("ERGO_HMR2_REPO", str(Path.home() / "hzhou" / "third_party" / "4D-Humans")),
        help="Optional path to a local 4D-Humans checkout.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("ERGO_HMR2_CHECKPOINT"),
        help="Optional HMR2 checkpoint path override.",
    )
    parser.add_argument(
        "--padding-scale",
        type=float,
        default=1.25,
        help="Expand the human-mask bounding box by this factor before HMR2 cropping.",
    )
    args = parser.parse_args()

    ensure_output_roots()

    clip_dir = resolve_clip_dir(args.clip_dir)
    clip_name = clip_dir.name
    rgb_path = clip_dir / f"{clip_name}.rgb.png"
    mask_path = clip_dir / "human_mask.npy"

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB sample not found: {rgb_path}")
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Human mask not found: {mask_path}. Run scripts/bash/run_human_mask.sh first."
        )

    hmr2_repo = Path(args.hmr2_repo).expanduser() if args.hmr2_repo else None
    stack = load_hmr2_modules(hmr2_repo)
    torch = stack["torch"]
    cv2 = stack["cv2"]

    image_rgb = np.array(Image.open(rgb_path).convert("RGB"))
    image_bgr = image_rgb[:, :, ::-1].copy()
    bbox_xyxy = load_mask_bbox(mask_path, image_rgb.shape[:2], padding_scale=args.padding_scale)

    checkpoint_path = args.checkpoint or stack["DEFAULT_CHECKPOINT"]
    stack["download_models"](stack["CACHE_DIR_4DHUMANS"])
    model, model_cfg = load_hmr2_checkpoint_compat(stack["load_hmr2"], checkpoint_path, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataset = stack["ViTDetDataset"](model_cfg, image_bgr, np.asarray([bbox_xyxy], dtype=np.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch = stack["recursive_to"](batch, device)

    with torch.no_grad():
        output = model(batch)

    pred_vertices = output["pred_vertices"][0].detach().cpu().numpy()
    pred_joints = output["pred_keypoints_3d"][0].detach().cpu().numpy()
    pred_cam_crop = output["pred_cam"][0].detach().cpu().numpy()
    pred_cam_t_crop = output["pred_cam_t"][0].detach().cpu().numpy()
    pred_smpl_params = {
        key: tensor_to_list(value[0])
        for key, value in output["pred_smpl_params"].items()
    }

    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

    from hmr2.utils.renderer import cam_crop_to_full  # Imported here after HMR2 becomes available.

    pred_cam_t_full = cam_crop_to_full(
        output["pred_cam"],
        box_center,
        box_size,
        img_size,
        scaled_focal_length,
    )[0].detach().cpu().numpy()

    vertices_camera = pred_vertices + pred_cam_t_full[None, :]
    joints_camera = pred_joints + pred_cam_t_full[None, :]
    faces = np.asarray(model.smpl.faces, dtype=np.int32)

    mesh_obj_path = clip_dir / "human_mesh.obj"
    vertices_path = clip_dir / "pred_vertices_3d.npy"
    joints_path = clip_dir / "pred_joints_3d.npy"
    smpl_json_path = clip_dir / "smpl_params.json"
    preview_path = clip_dir / "mesh_preview.png"
    stats_path = clip_dir / "mesh_recovery_stats.json"

    save_obj(mesh_obj_path, vertices_camera, faces)
    np.save(vertices_path, vertices_camera.astype(np.float32))
    np.save(joints_path, joints_camera.astype(np.float32))
    save_mesh_preview(image_rgb, vertices_camera, joints_camera, bbox_xyxy, preview_path)

    smpl_payload = {
        "clip_dir": str(clip_dir),
        "hmr2_checkpoint": str(checkpoint_path),
        "bbox_xyxy": bbox_xyxy.tolist(),
        "pred_cam_crop": pred_cam_crop.tolist(),
        "pred_cam_t_crop": pred_cam_t_crop.tolist(),
        "pred_cam_t_full": pred_cam_t_full.tolist(),
        "global_orient_rotmat": pred_smpl_params["global_orient"],
        "body_pose_rotmat": pred_smpl_params["body_pose"],
        "betas": pred_smpl_params["betas"],
    }
    write_json(smpl_json_path, smpl_payload)

    stats = {
        "status": "success",
        "clip_dir": str(clip_dir),
        "rgb_path": str(rgb_path),
        "mask_path": str(mask_path),
        "bbox_xyxy": bbox_xyxy.tolist(),
        "device": str(device),
        "hmr2_repo": str(hmr2_repo) if hmr2_repo else None,
        "hmr2_cache_dir": stack["CACHE_DIR_4DHUMANS"],
        "hmr2_checkpoint": str(checkpoint_path),
        "num_vertices": int(vertices_camera.shape[0]),
        "num_joints": int(joints_camera.shape[0]),
        "outputs": {
            "human_mesh_obj": str(mesh_obj_path),
            "pred_vertices_3d_npy": str(vertices_path),
            "pred_joints_3d_npy": str(joints_path),
            "smpl_params_json": str(smpl_json_path),
            "mesh_preview_png": str(preview_path),
            "mesh_recovery_stats_json": str(stats_path),
        },
        "notes": [
            "This stage uses the existing human mask to define the HMR2 crop.",
            "pred_vertices_3d.npy and pred_joints_3d.npy are saved in camera-space coordinates.",
        ],
    }
    write_json(stats_path, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
