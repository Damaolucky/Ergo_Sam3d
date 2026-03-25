#!/usr/bin/env bash
# Run a coarse mesh-to-pointcloud alignment stage for one clip output folder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/run_align_mesh.sh <clip_output_folder_name_or_path>

Example:
  bash scripts/bash/run_align_mesh.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4

Notes:
  This is a coarse PCA-based alignment scaffold for the next stage.
  It expects human_mesh.obj from mesh recovery and human_pointcloud.npy from the verified depth pipeline.
EOF
}


CLIP_DIR_NAME="${1:-}"
if [ -z "$CLIP_DIR_NAME" ]; then
  usage
  exit 1
fi

PY_SCRIPT="$(repo_python_script "align_mesh_to_pointcloud.py")"
require_file "$PY_SCRIPT" "Python script"
PYTHON_BIN="$(choose_python_bin)"

"$PYTHON_BIN" "$PY_SCRIPT" "$CLIP_DIR_NAME"
