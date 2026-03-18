#!/usr/bin/env bash
# Run YOLO segmentation on the RGB sample and build a human-only point cloud.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/run_human_mask.sh <clip_output_folder_name_or_path>

Example:
  bash scripts/bash/run_human_mask.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4

Environment:
  ERGO_WORK_ROOT  Default: ~/hzhou. Controls local outputs/.
  ERGO_YOLO_MODEL Default: yolov8n-seg.pt.
EOF
}


CLIP_DIR_NAME="${1:-}"
if [ -z "$CLIP_DIR_NAME" ]; then
  usage
  exit 1
fi

PY_SCRIPT="$(repo_python_script "generate_human_mask.py")"
require_file "$PY_SCRIPT" "Python script"
PYTHON_BIN="$(choose_python_bin)"

"$PYTHON_BIN" "$PY_SCRIPT" "$CLIP_DIR_NAME"
