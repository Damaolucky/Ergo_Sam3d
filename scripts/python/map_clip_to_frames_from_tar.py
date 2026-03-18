#!/usr/bin/env python3
"""Map one annotated clip to RGB frame indices and its nearest depth frame."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_utils import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_JSON_ROOT,
    DEFAULT_OUTPUTS_DIR,
    ensure_output_roots,
    load_json,
    write_json,
)


CAMERAS = ("angled", "overhead", "sagittal")


def nearest_index(arr: np.ndarray, value: float) -> int:
    """Return the index whose value is closest to the target."""
    if arr.size == 0:
        raise ValueError("Empty timestamp array.")
    return int(np.argmin(np.abs(arr - value)))


def infer_camera(source_video: str) -> str:
    """Infer the camera name from the source video stem."""
    stem = Path(source_video).stem.lower()
    for camera in CAMERAS:
        if stem.endswith(camera):
            return camera
    raise ValueError(f"Could not infer camera from source_video: {source_video}")


def resolve_json_path(session_name: str, json_override: str | None) -> Path:
    """Resolve the session annotation JSON, allowing a manual override."""
    if json_override:
        path = Path(json_override).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {path}")
        return path

    exact = DEFAULT_JSON_ROOT / f"{session_name.replace('-', '_')}.json"
    if exact.exists():
        return exact

    prefix = session_name.replace("-", "_")
    matches = sorted(DEFAULT_JSON_ROOT.glob(f"{prefix}*.json"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find JSON under {DEFAULT_JSON_ROOT} for session {session_name}"
    )


def load_npy_from_tar_member(tf: tarfile.TarFile, member_name: str) -> np.ndarray:
    """Load one `.npy` member from the session tarball."""
    member = tf.getmember(member_name)
    handle = tf.extractfile(member)
    if handle is None:
        raise FileNotFoundError(f"Could not extract tar member: {member_name}")
    return np.load(io.BytesIO(handle.read()), allow_pickle=True)


def session_cache_dir(session_name: str) -> Path:
    """Return the local cache directory for one session."""
    return DEFAULT_CACHE_ROOT / session_name


def cache_paths(session_name: str) -> dict[str, Path]:
    """Return cache file locations for one session."""
    base = session_cache_dir(session_name)
    return {"base": base, "index": base / "index.json"}


def detect_time_unit_factor(ts: np.ndarray) -> tuple[float, str]:
    """Infer the timestamp unit and return the factor needed to reach seconds."""
    if ts.size == 0:
        return 1.0, "empty"

    arr = np.asarray(ts, dtype=np.float64).reshape(-1)
    max_val = float(np.max(np.abs(arr)))

    if max_val > 1e17:
        return 1e-9, "nanoseconds->seconds"
    if max_val > 1e14:
        return 1e-6, "microseconds->seconds"
    if max_val > 1e11:
        return 1e-3, "milliseconds->seconds"
    return 1.0, "seconds"


def normalize_to_relative_seconds(ts: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Convert raw timestamps to relative seconds starting at the first frame."""
    arr = np.asarray(ts, dtype=np.float64).reshape(-1)
    factor, unit_label = detect_time_unit_factor(arr)
    arr_seconds = arr * factor
    t0 = float(arr_seconds[0]) if arr_seconds.size > 0 else 0.0
    return arr_seconds - t0, unit_label, t0


def build_or_load_index(
    session_name: str,
    tar_path: Path,
    target_cam: str,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Build a lightweight cache for the requested camera only."""
    paths = cache_paths(session_name)
    paths["base"].mkdir(parents=True, exist_ok=True)

    existing_index: dict[str, Any] = {
        "session": session_name,
        "tar_path": str(tar_path),
        "cameras": {},
    }
    if paths["index"].exists():
        existing_index = load_json(paths["index"])

    if (
        not force_reindex
        and target_cam in existing_index.get("cameras", {})
    ):
        cam_info = existing_index["cameras"][target_cam]
        color_cache = Path(cam_info["color_timestamp_cache"])
        depth_cache = Path(cam_info["depth_timestamp_cache"])
        if color_cache.exists() and depth_cache.exists():
            return existing_index

    print(f"[cache] building cache only for camera={target_cam} in session={session_name} ...")

    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        color_member = None
        depth_member = None
        depth_members: list[str] = []

        for member in members:
            name = member.name
            if not member.isfile() or f"/{target_cam}/" not in name:
                continue
            if name.endswith("color.timestamps.npy"):
                color_member = name
            elif name.endswith("depth.timestamps.npy"):
                depth_member = name
            elif name.endswith(".depth.image.npy"):
                depth_members.append(name)

        if color_member is None or depth_member is None:
            raise FileNotFoundError(
                f"Could not find timestamp files for camera={target_cam} in {tar_path}"
            )

        color_cache = paths["base"] / f"{target_cam}.color.timestamps.npy"
        depth_cache = paths["base"] / f"{target_cam}.depth.timestamps.npy"

        if force_reindex or not color_cache.exists():
            np.save(color_cache, load_npy_from_tar_member(tf, color_member))
        if force_reindex or not depth_cache.exists():
            np.save(depth_cache, load_npy_from_tar_member(tf, depth_member))

    existing_index["session"] = session_name
    existing_index["tar_path"] = str(tar_path)
    existing_index.setdefault("cameras", {})
    existing_index["cameras"][target_cam] = {
        "color_timestamp_member": color_member,
        "depth_timestamp_member": depth_member,
        "color_timestamp_cache": str(color_cache),
        "depth_timestamp_cache": str(depth_cache),
        "depth_members": sorted(depth_members),
    }
    write_json(paths["index"], existing_index)

    print(f"[cache] saved/updated index: {paths['index']}")
    return existing_index


def build_result(
    session_name: str,
    meta: dict[str, Any],
    tar_path: Path,
    index: dict[str, Any],
) -> dict[str, Any]:
    """Build the final mapping payload for one clip."""
    source_video = meta["source_video"]
    start_t = float(meta["source_start"])
    end_t = float(meta["source_end"])
    mid_t = (start_t + end_t) / 2.0

    camera = infer_camera(source_video)
    if camera not in index["cameras"]:
        raise KeyError(f"Camera {camera} not found in cached index for session {session_name}")

    cam_info = index["cameras"][camera]
    color_ts_raw = np.asarray(
        np.load(cam_info["color_timestamp_cache"], allow_pickle=True)
    ).reshape(-1)
    depth_ts_raw = np.asarray(
        np.load(cam_info["depth_timestamp_cache"], allow_pickle=True)
    ).reshape(-1)

    # Verified fix: compare clip times against timestamps normalized to relative seconds.
    color_ts_rel, color_unit_label, color_t0 = normalize_to_relative_seconds(color_ts_raw)
    depth_ts_rel, depth_unit_label, depth_t0 = normalize_to_relative_seconds(depth_ts_raw)

    start_color_idx = nearest_index(color_ts_rel, start_t)
    end_color_idx = nearest_index(color_ts_rel, end_t)
    mid_color_idx = nearest_index(color_ts_rel, mid_t)

    target_depth_time = float(color_ts_rel[mid_color_idx])
    mid_depth_idx = nearest_index(depth_ts_rel, target_depth_time)

    depth_member_name = None
    depth_members = cam_info["depth_members"]
    if depth_members:
        if len(depth_members) == len(depth_ts_rel):
            depth_member_name = depth_members[mid_depth_idx]
        elif mid_depth_idx < len(depth_members):
            depth_member_name = depth_members[mid_depth_idx]

    approx_color_fps = None
    approx_depth_fps = None
    if color_ts_rel.size > 1:
        color_dt = np.diff(color_ts_rel)
        color_dt = color_dt[color_dt > 0]
        if color_dt.size > 0:
            approx_color_fps = float(1.0 / np.median(color_dt))
    if depth_ts_rel.size > 1:
        depth_dt = np.diff(depth_ts_rel)
        depth_dt = depth_dt[depth_dt > 0]
        if depth_dt.size > 0:
            approx_depth_fps = float(1.0 / np.median(depth_dt))

    return {
        "camera": camera,
        "tar_path": str(tar_path),
        "cache_dir": str(session_cache_dir(session_name)),
        "source_video": source_video,
        "source_start": start_t,
        "source_end": end_t,
        "source_duration": float(meta["source_duration"]),
        "color_timestamp_member": cam_info["color_timestamp_member"],
        "depth_timestamp_member": cam_info["depth_timestamp_member"],
        "color_timestamp_cache": cam_info["color_timestamp_cache"],
        "depth_timestamp_cache": cam_info["depth_timestamp_cache"],
        "color_timestamp_unit_guess": color_unit_label,
        "depth_timestamp_unit_guess": depth_unit_label,
        "color_timestamp_t0_seconds": color_t0,
        "depth_timestamp_t0_seconds": depth_t0,
        "approx_color_fps": approx_color_fps,
        "approx_depth_fps": approx_depth_fps,
        "color_frame_range": [start_color_idx, end_color_idx],
        "mid_color_frame": {
            "index": mid_color_idx,
            "timestamp_relative_seconds": float(color_ts_rel[mid_color_idx]),
        },
        "nearest_depth_frame": {
            "index": mid_depth_idx,
            "timestamp_relative_seconds": float(depth_ts_rel[mid_depth_idx]),
            "tar_member": depth_member_name,
        },
        "metadata_fields": {
            "height1": meta.get("height1"),
            "height1_strength": meta.get("height1_strength"),
            "height2": meta.get("height2"),
            "height2_strength": meta.get("height2_strength"),
            "weight": meta.get("weight"),
            "ratio": meta.get("ratio"),
            "take": meta.get("take"),
        },
    }


def main() -> None:
    """Parse arguments and write the mapping JSON for one clip."""
    parser = argparse.ArgumentParser(
        description=(
            "Map one clip to an RGB frame range and a nearest depth frame using "
            "the session tarball plus a small local cache."
        )
    )
    parser.add_argument("--session", required=True, help="Session name, e.g. 2024-05-03_15")
    parser.add_argument("--clip-name", required=True, help="Full clip filename key in JSON.")
    parser.add_argument("--json", default=None, help="Optional annotation JSON override.")
    parser.add_argument("--tar-path", default=None, help="Optional session tar.gz override.")
    parser.add_argument("--save", default=None, help="Optional output path.")
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild the timestamp cache for this session/camera.",
    )
    args = parser.parse_args()

    ensure_output_roots()

    session_name = args.session
    tar_path = Path(args.tar_path).expanduser() if args.tar_path else DEFAULT_DATA_ROOT / f"{session_name}.tar.gz"
    json_path = resolve_json_path(session_name, args.json)
    save_path = (
        Path(args.save).expanduser()
        if args.save
        else DEFAULT_OUTPUTS_DIR / f"{args.clip_name}.mapping.json"
    )

    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")

    clip_meta = load_json(json_path)
    if args.clip_name not in clip_meta:
        examples = list(clip_meta.keys())[:5]
        raise KeyError(
            f"Clip not found in JSON: {args.clip_name}\nSample keys:\n" + "\n".join(examples)
        )

    target_cam = infer_camera(clip_meta[args.clip_name]["source_video"])
    index = build_or_load_index(
        session_name,
        tar_path,
        target_cam=target_cam,
        force_reindex=args.force_reindex,
    )

    result = build_result(session_name, clip_meta[args.clip_name], tar_path, index)
    result["clip_name"] = args.clip_name
    result["json_path"] = str(json_path)

    pretty = json.dumps(result, indent=2)
    print(pretty)
    write_json(save_path, result)
    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    main()
