#!/usr/bin/env bash
# Run clip-to-frame/depth mapping for a single session clip.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/run_clip_mapping.sh <session_name> <clip_name> [--force-reindex]

Example:
  bash scripts/bash/run_clip_mapping.sh 2024-05-03_15 2024_05_03_15_sagittal_high_24_high_24_5_3_1_lift.mp4

Environment:
  ERGO_WORK_ROOT  Default: ~/hzhou. Controls local outputs/ and cache/.
EOF
}


SESSION_NAME="${1:-}"
CLIP_NAME="${2:-}"
EXTRA_FLAG="${3:-}"

if [ -z "$SESSION_NAME" ] || [ -z "$CLIP_NAME" ]; then
  usage
  exit 1
fi

PY_SCRIPT="$(repo_python_script "map_clip_to_frames_from_tar.py")"
require_file "$PY_SCRIPT" "Python script"
PYTHON_BIN="$(choose_python_bin)"

mkdir -p "$ERGO_WORK_ROOT/outputs"

CMD=(
  "$PYTHON_BIN" "$PY_SCRIPT"
  --session "$SESSION_NAME"
  --clip-name "$CLIP_NAME"
)

if [ "$EXTRA_FLAG" = "--force-reindex" ]; then
  CMD+=(--force-reindex)
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"
