#!/usr/bin/env bash
# Move extracted sample assets into a clip folder and build scene geometry files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/run_prepare_geometry.sh <sample_manifest_json_filename_or_path>

Example:
  bash scripts/bash/run_prepare_geometry.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4.sample_manifest.json

Environment:
  ERGO_WORK_ROOT  Default: ~/hzhou. Controls local outputs/.
EOF
}


MANIFEST_NAME="${1:-}"
if [ -z "$MANIFEST_NAME" ]; then
  usage
  exit 1
fi

PY_SCRIPT="$(repo_python_script "prepare_geometry_sample.py")"
require_file "$PY_SCRIPT" "Python script"
PYTHON_BIN="$(choose_python_bin)"

"$PYTHON_BIN" "$PY_SCRIPT" "$MANIFEST_NAME"
