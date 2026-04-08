#!/usr/bin/env bash
# Estimate the final-frame destination shelf/object height for one clip output folder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/run_estimate_shelf_height.sh <clip_output_folder_name_or_path> [extra_python_args...]

Example:
  bash scripts/bash/run_estimate_shelf_height.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4__last_high_24
  bash scripts/bash/run_estimate_shelf_height.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4__last_high_24 --level high --shelf-side right

Notes:
  This stage assumes the clip folder is the final-frame sample folder.
  The estimate uses aligned_mesh_vertices.npy as the floor/human reference when available.
EOF
}


CLIP_DIR_NAME="${1:-}"
if [ -z "$CLIP_DIR_NAME" ]; then
  usage
  exit 1
fi
shift

PY_SCRIPT="$(repo_python_script "estimate_shelf_height.py")"
require_file "$PY_SCRIPT" "Python script"
PYTHON_BIN="$(choose_python_bin)"

"$PYTHON_BIN" "$PY_SCRIPT" "$CLIP_DIR_NAME" "$@"
