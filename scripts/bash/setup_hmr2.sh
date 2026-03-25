#!/usr/bin/env bash
# Clone and install the external 4D-Humans/HMR2 dependency used by the mesh stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"


usage() {
  cat <<'EOF'
Usage:
  bash scripts/bash/setup_hmr2.sh

Environment:
  ERGO_HMR2_ENV_NAME     Default: hzhou
  ERGO_HMR2_REPO         Default: ~/hzhou/third_party/4D-Humans
  ERGO_HMR2_SMPL_SOURCE  Optional local path to the official SMPL neutral model

What this does:
  1. Clones shubham-goel/4D-Humans into the configured repo path.
  2. Installs HMR2 into the selected conda environment with `pip install -e`.
  3. Optionally copies the official SMPL model into ~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl.

What this does not do:
  It does not bypass SMPL licensing. You still need the official neutral model file.
EOF
}


if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

ERGO_HMR2_ENV_NAME="${ERGO_HMR2_ENV_NAME:-hzhou}"
ERGO_HMR2_REPO="${ERGO_HMR2_REPO:-$ERGO_WORK_ROOT/third_party/4D-Humans}"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda command not found."
  exit 1
fi

mkdir -p "$(dirname "$ERGO_HMR2_REPO")"

if [ ! -d "$ERGO_HMR2_REPO/.git" ]; then
  git clone https://github.com/shubham-goel/4D-Humans.git "$ERGO_HMR2_REPO"
else
  echo "Repo already exists: $ERGO_HMR2_REPO"
fi

conda run -n "$ERGO_HMR2_ENV_NAME" python -m pip install --upgrade pip setuptools wheel
conda run -n "$ERGO_HMR2_ENV_NAME" python -m pip install -e "$ERGO_HMR2_REPO"

if [ -n "${ERGO_HMR2_SMPL_SOURCE:-}" ]; then
  if [ ! -f "$ERGO_HMR2_SMPL_SOURCE" ]; then
    echo "Error: ERGO_HMR2_SMPL_SOURCE does not exist:"
    echo "  $ERGO_HMR2_SMPL_SOURCE"
    exit 1
  fi
  mkdir -p "$HOME/.cache/4DHumans/data/smpl"
  cp -f "$ERGO_HMR2_SMPL_SOURCE" "$HOME/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl"
  echo "Copied SMPL model into ~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl"
fi

echo
echo "Setup finished."
echo "Recommended runtime env:"
echo "  export ERGO_HMR2_PYTHON=\"$HOME/miniconda3/envs/$ERGO_HMR2_ENV_NAME/bin/python\""
echo "  export ERGO_HMR2_REPO=\"$ERGO_HMR2_REPO\""
echo
echo "If you still need the SMPL neutral model, obtain it from the official SMPL site and then rerun:"
echo "  ERGO_HMR2_SMPL_SOURCE=/path/to/SMPL_NEUTRAL.pkl bash scripts/bash/setup_hmr2.sh"
