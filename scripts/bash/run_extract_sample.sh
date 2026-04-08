#!/usr/bin/env bash
# Extract the action-aware keyframe RGB/depth sample from a mapping JSON.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/run_extract_sample.sh <mapping_json_filename_or_path> [extra_python_args...]

Example:
  bash scripts/bash/run_extract_sample.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4.mapping.json
  bash scripts/bash/run_extract_sample.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4.mapping.json --sample-roles first
  bash scripts/bash/run_extract_sample.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_put.mp4.mapping.json --sample-roles last

Default:
  --sample-roles auto
  *_lift.mp4 -> first frame at the source shelf/object position
  *_put.mp4  -> last frame at the destination shelf/object position

Environment:
  ERGO_WORK_ROOT  Default: ~/hzhou. Controls local outputs/.
EOF
}


MAPPING_NAME="${1:-}"
if [ -z "$MAPPING_NAME" ]; then
  usage
  exit 1
fi
shift

PY_SCRIPT="$(repo_python_script "extract_sample_from_mapping.py")"
require_file "$PY_SCRIPT" "Python script"
PYTHON_BIN="$(choose_python_bin)"

"$PYTHON_BIN" "$PY_SCRIPT" "$MAPPING_NAME" "$@"
