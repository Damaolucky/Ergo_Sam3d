#!/usr/bin/env bash
# Run the optional SAM3D trial once the external repo and checkpoints are available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/run_sam3d_trial.sh <clip_output_folder_name_or_path>

Example:
  bash scripts/bash/run_sam3d_trial.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4

Notes:
  This stage is scaffolded only. The known blocker is Hugging Face access to
  facebook/sam-3d-body-dinov3, not a bug in the local pipeline logic.

Environment:
  ERGO_WORK_ROOT  Default: ~/hzhou.
EOF
}


CLIP_DIR_NAME="${1:-}"
if [ -z "$CLIP_DIR_NAME" ]; then
  usage
  exit 1
fi

OUTPUTS_ROOT="$ERGO_WORK_ROOT/outputs"
REPO_DIR="$ERGO_WORK_ROOT/sam-3d-body"
CHECKPOINT_DIR="$REPO_DIR/checkpoints/sam-3d-body-dinov3"
CHECKPOINT_PATH="$CHECKPOINT_DIR/model.ckpt"
MHR_PATH="$CHECKPOINT_DIR/assets/mhr_model.pt"
PYTHON_BIN="$(choose_python_bin)"

if [ -d "$CLIP_DIR_NAME" ]; then
  CLIP_DIR="$CLIP_DIR_NAME"
elif [ -d "$OUTPUTS_ROOT/$CLIP_DIR_NAME" ]; then
  CLIP_DIR="$OUTPUTS_ROOT/$CLIP_DIR_NAME"
else
  echo "Error: clip dir not found:"
  echo "  $CLIP_DIR_NAME"
  echo "  $OUTPUTS_ROOT/$CLIP_DIR_NAME"
  exit 1
fi

RGB_PATH="$CLIP_DIR/$(basename "$CLIP_DIR").rgb.png"
require_file "$RGB_PATH" "RGB file"

if [ ! -d "$REPO_DIR" ]; then
  echo "Error: repo not found:"
  echo "  $REPO_DIR"
  echo
  echo "Clone it with:"
  echo "  bash scripts/bash/setup_sam3d_body.sh"
  exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ] || [ ! -f "$MHR_PATH" ]; then
  echo "Error: SAM3D checkpoints not found."
  echo "Expected:"
  echo "  $CHECKPOINT_PATH"
  echo "  $MHR_PATH"
  echo
  echo "This usually means model access has not been approved yet."
  echo "Official quick start says to download with:"
  echo "  hf download facebook/sam-3d-body-dinov3 --local-dir $CHECKPOINT_DIR"
  exit 1
fi

INPUT_DIR="$CLIP_DIR/sam3d_input"
SAM3D_OUT="$CLIP_DIR/sam3d_output"
mkdir -p "$INPUT_DIR" "$SAM3D_OUT"

cp -f "$RGB_PATH" "$INPUT_DIR/"

cd "$REPO_DIR"

"$PYTHON_BIN" demo.py \
  --image_folder "$INPUT_DIR" \
  --output_folder "$SAM3D_OUT" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --mhr_path "$MHR_PATH"

echo
echo "SAM3D trial finished."
echo "Outputs should be under:"
echo "  $SAM3D_OUT"
