#!/usr/bin/env bash
# Run the production keyframe-to-shelf-height pipeline for one annotated clip.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/run_keyframe_pipeline.sh <session> <clip_filename>

Example:
  bash scripts/bash/run_keyframe_pipeline.sh \
    2024-05-03_15 \
    2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4

Keyframe rule:
  *_lift.mp4 -> first frame at height1/source position
  *_put.mp4  -> last frame at height2/destination position

Optional environment variables:
  ERGO_WORK_ROOT              Default: ~/hzhou. Controls outputs/.
  ERGO_TARGET_HUMAN_HEIGHT_M  If set, passed to run_align_mesh.sh.
  ERGO_KNOWN_HUMAN_HEIGHT_M   If set, passed to run_estimate_shelf_height.sh.
  ERGO_SHELF_SIDE             Optional shelf side: left, right, or auto.
  ERGO_LEVEL                  Optional target level override: high, mid, or low.
  ERGO_CALIBRATION_JSON       Optional camera calibration JSON for geometric mask reprojection.
EOF
}


SESSION_NAME="${1:-}"
CLIP_NAME="${2:-}"
if [ -z "$SESSION_NAME" ] || [ -z "$CLIP_NAME" ]; then
  usage
  exit 1
fi

PYTHON_BIN="$(choose_python_bin)"
EXTRACT_PAYLOAD="$(mktemp)"
trap 'rm -f "$EXTRACT_PAYLOAD"' EXIT

bash "$SCRIPT_DIR/run_clip_mapping.sh" "$SESSION_NAME" "$CLIP_NAME"
bash "$SCRIPT_DIR/run_extract_sample.sh" "$CLIP_NAME.mapping.json" > "$EXTRACT_PAYLOAD"

KEY_SAMPLE="$("$PYTHON_BIN" - "$EXTRACT_PAYLOAD" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
samples = payload.get("generated_samples") or []
if len(samples) != 1:
    raise SystemExit(f"Expected exactly one keyframe sample, found {len(samples)}.")
print(samples[0]["sample_name"])
PY
)"

echo "Keyframe sample: $KEY_SAMPLE"

bash "$SCRIPT_DIR/run_prepare_geometry.sh" "${KEY_SAMPLE}.sample_manifest.json"
bash "$SCRIPT_DIR/run_human_mask.sh" "$KEY_SAMPLE"
bash "$SCRIPT_DIR/run_human_mesh_recovery.sh" "$KEY_SAMPLE"

ALIGN_ARGS=()
if [ -n "${ERGO_TARGET_HUMAN_HEIGHT_M:-}" ]; then
  ALIGN_ARGS+=(--target-human-height-m "$ERGO_TARGET_HUMAN_HEIGHT_M")
fi
bash "$SCRIPT_DIR/run_align_mesh.sh" "$KEY_SAMPLE" "${ALIGN_ARGS[@]}"

SHELF_ARGS=()
if [ -n "${ERGO_KNOWN_HUMAN_HEIGHT_M:-}" ]; then
  SHELF_ARGS+=(--known-human-height-m "$ERGO_KNOWN_HUMAN_HEIGHT_M")
fi
if [ -n "${ERGO_SHELF_SIDE:-}" ]; then
  SHELF_ARGS+=(--shelf-side "$ERGO_SHELF_SIDE")
fi
if [ -n "${ERGO_LEVEL:-}" ]; then
  SHELF_ARGS+=(--level "$ERGO_LEVEL")
fi
bash "$SCRIPT_DIR/run_estimate_shelf_height.sh" "$KEY_SAMPLE" "${SHELF_ARGS[@]}"

echo "Pipeline complete: ${ERGO_WORK_ROOT:-$HOME/hzhou}/outputs/$KEY_SAMPLE"
