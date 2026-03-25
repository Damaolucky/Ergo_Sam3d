#!/usr/bin/env python3
"""Extract endpoint RGB/depth samples from a mapping JSON."""

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
    build_sample_output_name,
    ensure_output_roots,
    format_position_label,
    load_json,
    resolve_in_outputs,
    save_depth_vis,
    save_pickle,
    write_json,
)


def load_npy_from_bytes(payload: bytes) -> np.ndarray:
    """Load one NumPy array from raw tar bytes."""
    return np.load(io.BytesIO(payload), allow_pickle=True)


def load_pickle_from_bytes(payload: bytes) -> Any:
    """Load one pickle object from raw tar bytes."""
    return pickle.loads(payload)


def collect_required_tar_bytes(
    tar_path: Path,
    *,
    exact_members: dict[str, str],
    suffix_members: dict[str, str],
) -> dict[str, bytes]:
    """Stream a tar.gz once and collect only the required members."""
    remaining_exact = dict(exact_members)
    remaining_suffix = dict(suffix_members)
    found: dict[str, bytes] = {}

    with tarfile.open(tar_path, "r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue

            alias = None
            if member.name in remaining_exact.values():
                for key, exact_name in list(remaining_exact.items()):
                    if member.name == exact_name:
                        alias = key
                        del remaining_exact[key]
                        break
            else:
                for key, suffix in list(remaining_suffix.items()):
                    if member.name.endswith(suffix):
                        alias = key
                        del remaining_suffix[key]
                        break

            if alias is None:
                continue

            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(f"Could not extract tar member: {member.name}")
            found[alias] = handle.read()

            if not remaining_exact and not remaining_suffix:
                break

    missing = list(remaining_exact.keys()) + list(remaining_suffix.keys())
    if missing:
        raise FileNotFoundError(f"Missing required tar members: {missing}")
    return found


def ensure_ffmpeg() -> str:
    """Find an ffmpeg binary that can extract representative RGB frames."""
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


def extract_rgb_from_clip_time(
    clip_path: Path,
    clip_time_sec: float,
    duration_sec: float,
    out_png: Path,
    *,
    fps_hint: float | None,
) -> None:
    """Extract one RGB frame from the clip video at a chosen clip-relative time."""
    ffmpeg_bin = ensure_ffmpeg()
    frame_guard = 1.0 / max(float(fps_hint or 30.0), 1.0)
    max_seek = max(duration_sec - frame_guard, 0.0)
    sample_time = min(max(clip_time_sec, 0.0), max_seek)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{sample_time:.6f}",
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


def legacy_sample_frames(mapping: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a single midpoint sample entry for older mapping files."""
    clip_name = mapping["clip_name"]
    meta = mapping.get("metadata_fields", {})
    mid_label = format_position_label(
        meta.get("height1"),
        meta.get("height1_strength"),
        fallback="mid",
    )
    return {
        "mid": {
            "sample_role": "mid",
            "position_label": mid_label,
            "sample_name": build_sample_output_name(clip_name, "mid", mid_label),
            "clip_time_seconds": float(mapping["source_duration"]) / 2.0,
            "color_frame": mapping["mid_color_frame"],
            "nearest_depth_frame": mapping["nearest_depth_frame"],
        }
    }


def choose_sample_frames(mapping: dict[str, Any], sample_roles: list[str]) -> list[dict[str, Any]]:
    """Select the requested sample frame specs from the mapping payload."""
    sample_frames = mapping.get("sample_frames") or legacy_sample_frames(mapping)
    selected: list[dict[str, Any]] = []
    for role in sample_roles:
        if role not in sample_frames:
            raise KeyError(f"Requested sample role '{role}' not present in mapping JSON.")
        selected.append(sample_frames[role])
    return selected


def save_one_sample(
    *,
    clip_name: str,
    mapping_path: Path,
    mapping: dict[str, Any],
    clip_path: Path,
    raw_depth_bytes: bytes,
    depth_scale: float,
    intrinsics: Any,
    sample_spec: dict[str, Any],
) -> dict[str, Any]:
    """Extract and persist one sample bundle for a chosen clip endpoint."""
    sample_name = sample_spec["sample_name"]

    rgb_png = DEFAULT_OUTPUTS_DIR / f"{sample_name}.rgb.png"
    depth_raw_npy = DEFAULT_OUTPUTS_DIR / f"{sample_name}.depth_raw.npy"
    depth_meters_npy = DEFAULT_OUTPUTS_DIR / f"{sample_name}.depth_meters.npy"
    depth_vis_png = DEFAULT_OUTPUTS_DIR / f"{sample_name}.depth_vis.png"
    intrinsics_pkl = DEFAULT_OUTPUTS_DIR / f"{sample_name}.intrinsics.pkl"
    manifest_json = DEFAULT_OUTPUTS_DIR / f"{sample_name}.sample_manifest.json"

    extract_rgb_from_clip_time(
        clip_path,
        float(sample_spec["clip_time_seconds"]),
        float(mapping["source_duration"]),
        rgb_png,
        fps_hint=mapping.get("approx_color_fps"),
    )

    depth_raw = np.asarray(load_npy_from_bytes(raw_depth_bytes))
    depth_meters = depth_raw.astype(np.float32) * depth_scale

    np.save(depth_raw_npy, depth_raw)
    np.save(depth_meters_npy, depth_meters)
    save_pickle(intrinsics_pkl, intrinsics)
    save_depth_vis(depth_meters, depth_vis_png)

    manifest = {
        "mapping_json": str(mapping_path),
        "clip_name": sample_name,
        "source_clip_name": clip_name,
        "sample_name": sample_name,
        "sample_role": sample_spec["sample_role"],
        "position_label": sample_spec["position_label"],
        "clip_video": str(clip_path),
        "camera": mapping["camera"],
        "tar_path": str(mapping["tar_path"]),
        "source_video": mapping["source_video"],
        "source_start": mapping["source_start"],
        "source_end": mapping["source_end"],
        "source_duration": mapping["source_duration"],
        "clip_time_seconds": sample_spec["clip_time_seconds"],
        "color_frame": sample_spec["color_frame"],
        "nearest_depth_frame": sample_spec["nearest_depth_frame"],
        "depth_scale": depth_scale,
        "outputs": {
            "rgb_png": str(rgb_png),
            "depth_raw_npy": str(depth_raw_npy),
            "depth_meters_npy": str(depth_meters_npy),
            "depth_vis_png": str(depth_vis_png),
            "intrinsics_pkl": str(intrinsics_pkl),
            "sample_manifest_json": str(manifest_json),
        },
    }
    write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    """Parse arguments and extract endpoint sample assets for one mapped clip."""
    parser = argparse.ArgumentParser(
        description="Extract first/last RGB+depth samples from a mapping JSON."
    )
    parser.add_argument(
        "mapping",
        help=(
            "Mapping JSON filename or path, e.g. "
            "2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4.mapping.json"
        ),
    )
    parser.add_argument(
        "--sample-roles",
        nargs="+",
        default=["first", "last"],
        choices=["first", "last", "mid"],
        help="Which sample roles to extract. Default: first last",
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

    sample_specs = choose_sample_frames(mapping, args.sample_roles)

    exact_members = {
        f"depth::{sample_spec['sample_role']}": sample_spec["nearest_depth_frame"]["tar_member"]
        for sample_spec in sample_specs
    }
    suffix_members = {
        "depth_scale": f"/{camera}/depth.scale.npy",
        "intrinsics": f"/{camera}/depth.intrinsics.pkl",
    }
    raw_members = collect_required_tar_bytes(
        tar_path,
        exact_members=exact_members,
        suffix_members=suffix_members,
    )

    depth_scale = float(np.asarray(load_npy_from_bytes(raw_members["depth_scale"])).reshape(-1)[0])
    intrinsics = load_pickle_from_bytes(raw_members["intrinsics"])

    manifests: list[dict[str, Any]] = []
    for sample_spec in sample_specs:
        manifests.append(
            save_one_sample(
                clip_name=clip_name,
                mapping_path=mapping_path,
                mapping=mapping,
                clip_path=clip_path,
                raw_depth_bytes=raw_members[f"depth::{sample_spec['sample_role']}"],
                depth_scale=depth_scale,
                intrinsics=intrinsics,
                sample_spec=sample_spec,
            )
        )

    payload = {
        "mapping_json": str(mapping_path),
        "source_clip_name": clip_name,
        "sample_roles": args.sample_roles,
        "generated_samples": manifests,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
