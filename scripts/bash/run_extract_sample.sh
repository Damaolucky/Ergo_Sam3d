#!/usr/bin/env bash
# Extract one RGB/depth sample pair from a mapping JSON.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/run_extract_sample.sh <mapping_json_filename_or_path>

Example:
  bash scripts/bash/run_extract_sample.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4.mapping.json

Environment:
  ERGO_WORK_ROOT  Default: ~/hzhou. Controls local outputs/.
EOF
}


MAPPING_NAME="${1:-}"
if [ -z "$MAPPING_NAME" ]; then
  usage
  exit 1
fi

PY_SCRIPT="$(repo_python_script "extract_sample_from_mapping.py")"
require_file "$PY_SCRIPT" "Python script"
PYTHON_BIN="$(choose_python_bin)"

"$PYTHON_BIN" "$PY_SCRIPT" "$MAPPING_NAME"
