#!/usr/bin/env bash
# Run HMR2-based human mesh recovery for one clip output folder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/run_human_mesh_recovery.sh <clip_output_folder_name_or_path>

Example:
  bash scripts/bash/run_human_mesh_recovery.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4

Environment:
  ERGO_HMR2_PYTHON      Optional Python binary with HMR2 dependencies installed.
  ERGO_HMR2_REPO        Default: ~/hzhou/third_party/4D-Humans
  ERGO_HMR2_CHECKPOINT  Optional HMR2 checkpoint override.
  ERGO_OPENGL_PLATFORM  Default: egl

Notes:
  This stage uses the verified human mask to define the HMR2 crop.
  HMR2 still requires the official SMPL neutral model file.
EOF
}


CLIP_DIR_NAME="${1:-}"
if [ -z "$CLIP_DIR_NAME" ]; then
  usage
  exit 1
fi

PY_SCRIPT="$(repo_python_script "recover_human_mesh.py")"
require_file "$PY_SCRIPT" "Python script"

if [ -n "${ERGO_HMR2_PYTHON:-}" ]; then
  PYTHON_BIN="$ERGO_HMR2_PYTHON"
else
  PYTHON_BIN="$(choose_python_bin)"
fi

"$PYTHON_BIN" "$PY_SCRIPT" "$CLIP_DIR_NAME"
