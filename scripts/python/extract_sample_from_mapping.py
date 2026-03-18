#!/usr/bin/env python3
"""Extract one RGB sample and one aligned depth sample from a mapping JSON."""

from __future__ import annotations

import argparse
import io
import json
import pickle
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_utils import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_OUTPUTS_DIR,
    ensure_output_roots,
    load_json,
    resolve_in_outputs,
    save_depth_vis,
    save_pickle,
    write_json,
)


def extract_tar_bytes(tf: tarfile.TarFile, member_name: str) -> bytes:
    """Extract raw bytes for one tar member."""
    member = tf.getmember(member_name)
    handle = tf.extractfile(member)
    if handle is None:
        raise FileNotFoundError(f"Could not extract tar member: {member_name}")
    return handle.read()


def load_npy_from_tar(tf: tarfile.TarFile, member_name: str) -> np.ndarray:
    """Load one NumPy array from the tarball."""
    return np.load(io.BytesIO(extract_tar_bytes(tf, member_name)), allow_pickle=True)


def load_pickle_from_tar(tf: tarfile.TarFile, member_name: str) -> Any:
    """Load one pickle object from the tarball."""
    return pickle.loads(extract_tar_bytes(tf, member_name))


def find_member_by_suffix(tf: tarfile.TarFile, suffix: str) -> str:
    """Find the shortest tar member path that matches a suffix."""
    matches = [member.name for member in tf.getmembers() if member.isfile() and member.name.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"Could not find member ending with: {suffix}")
    return sorted(matches, key=len)[0]


def ensure_ffmpeg() -> str:
    """Find an ffmpeg binary that can extract a representative RGB frame."""
    for name in ("ffmpeg", "/usr/bin/ffmpeg"):
        try:
            subprocess.run(
                [name, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return name
        except Exception:
            continue
    raise RuntimeError("ffmpeg not found in PATH.")


def extract_rgb_from_clip(clip_path: Path, duration_sec: float, out_png: Path) -> None:
    """Extract the clip midpoint frame as an RGB PNG."""
    ffmpeg_bin = ensure_ffmpeg()
    mid_sec = max(duration_sec / 2.0, 0.0)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{mid_sec:.6f}",
        "-i",
        str(clip_path),
        "-frames:v",
        "1",
        str(out_png),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def resolve_mapping_path(mapping_arg: str) -> Path:
    """Resolve a mapping JSON path directly or from the outputs root."""
    return resolve_in_outputs(mapping_arg, label="Mapping JSON")


def main() -> None:
    """Parse arguments and extract sample assets for one mapped clip."""
    parser = argparse.ArgumentParser(
        description="Extract one RGB sample frame and one depth sample from a mapping JSON."
    )
    parser.add_argument(
        "mapping",
        help=(
            "Mapping JSON filename or path, e.g. "
            "2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4.mapping.json"
        ),
    )
    args = parser.parse_args()

    ensure_output_roots()

    mapping_path = resolve_mapping_path(args.mapping)
    mapping = load_json(mapping_path)

    clip_name = mapping["clip_name"]
    tar_path = Path(mapping["tar_path"])
    camera = mapping["camera"]
    clip_path = DEFAULT_CLIPS_DIR / clip_name

    if not clip_path.exists():
        raise FileNotFoundError(f"Clip video not found: {clip_path}")
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")

    stem = clip_name
    rgb_png = DEFAULT_OUTPUTS_DIR / f"{stem}.rgb.png"
    depth_raw_npy = DEFAULT_OUTPUTS_DIR / f"{stem}.depth_raw.npy"
    depth_meters_npy = DEFAULT_OUTPUTS_DIR / f"{stem}.depth_meters.npy"
    depth_vis_png = DEFAULT_OUTPUTS_DIR / f"{stem}.depth_vis.png"
    intrinsics_pkl = DEFAULT_OUTPUTS_DIR / f"{stem}.intrinsics.pkl"
    manifest_json = DEFAULT_OUTPUTS_DIR / f"{stem}.sample_manifest.json"

    extract_rgb_from_clip(clip_path, float(mapping["source_duration"]), rgb_png)

    with tarfile.open(tar_path, "r:gz") as tf:
        depth_member = mapping["nearest_depth_frame"]["tar_member"]
        if not depth_member:
            raise RuntimeError("Mapping JSON has no nearest_depth_frame.tar_member")

        depth_raw = np.asarray(load_npy_from_tar(tf, depth_member))
        scale_member = find_member_by_suffix(tf, f"/{camera}/depth.scale.npy")
        intr_member = find_member_by_suffix(tf, f"/{camera}/depth.intrinsics.pkl")
        depth_scale = float(np.asarray(load_npy_from_tar(tf, scale_member)).reshape(-1)[0])
        intrinsics = load_pickle_from_tar(tf, intr_member)

    depth_meters = depth_raw.astype(np.float32) * depth_scale
    np.save(depth_raw_npy, depth_raw)
    np.save(depth_meters_npy, depth_meters)
    save_pickle(intrinsics_pkl, intrinsics)
    save_depth_vis(depth_meters, depth_vis_png)

    manifest = {
        "mapping_json": str(mapping_path),
        "clip_name": clip_name,
        "clip_video": str(clip_path),
        "camera": camera,
        "tar_path": str(tar_path),
        "source_video": mapping["source_video"],
        "source_start": mapping["source_start"],
        "source_end": mapping["source_end"],
        "source_duration": mapping["source_duration"],
        "mid_color_frame": mapping["mid_color_frame"],
        "nearest_depth_frame": mapping["nearest_depth_frame"],
        "depth_scale": depth_scale,
        "outputs": {
            "rgb_png": str(rgb_png),
            "depth_raw_npy": str(depth_raw_npy),
            "depth_meters_npy": str(depth_meters_npy),
            "depth_vis_png": str(depth_vis_png),
            "intrinsics_pkl": str(intrinsics_pkl),
        },
    }
    write_json(manifest_json, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
