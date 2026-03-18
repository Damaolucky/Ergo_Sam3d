#!/usr/bin/env bash
# Clone the optional SAM3D dependency and install the small helper dependency set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/setup_sam3d_body.sh

Description:
  Clone facebookresearch/sam-3d-body under ${ERGO_WORK_ROOT}/sam-3d-body if missing.
  This only prepares the external repo. Actual model download may still fail until
  Hugging Face access for facebook/sam-3d-body-dinov3 has been approved.

Environment:
  ERGO_WORK_ROOT  Default: ~/hzhou.
EOF
}


if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

REPO_DIR="$ERGO_WORK_ROOT/sam-3d-body"
PYTHON_BIN="$(choose_python_bin)"

mkdir -p "$ERGO_WORK_ROOT"
cd "$ERGO_WORK_ROOT"

if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/facebookresearch/sam-3d-body.git
else
  echo "Repo already exists: $REPO_DIR"
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install huggingface_hub

echo
echo "Next steps:"
echo "1) Read official install instructions:"
echo "   $REPO_DIR/INSTALL.md"
echo
echo "2) Download checkpoints after access is approved:"
echo "   hf download facebook/sam-3d-body-dinov3 --local-dir $REPO_DIR/checkpoints/sam-3d-body-dinov3"
echo
echo "3) Then run:"
echo "   bash scripts/bash/run_sam3d_trial.sh 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4"
