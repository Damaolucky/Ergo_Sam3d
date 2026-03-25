#!/usr/bin/env bash
# Shared helpers for bash entrypoints in this repository.

set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$COMMON_DIR/../.." && pwd)"
export ERGO_WORK_ROOT="${ERGO_WORK_ROOT:-$HOME/hzhou}"


choose_python_bin() {
  if [ -n "${ERGO_PYTHON:-}" ]; then
    if [ -x "$ERGO_PYTHON" ] || command -v "$ERGO_PYTHON" >/dev/null 2>&1; then
      printf '%s\n' "$ERGO_PYTHON"
      return 0
    fi

    echo "Error: ERGO_PYTHON is set but not executable:"
    echo "  $ERGO_PYTHON"
    return 1
  fi

  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "python"
    return 0
  fi

  echo "Error: neither python3 nor python was found in PATH." >&2
  return 1
}


repo_python_script() {
  local script_name="$1"
  printf '%s\n' "$REPO_ROOT/scripts/python/$script_name"
}


require_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "$path" ]; then
    echo "Error: $label not found:"
    echo "  $path"
    exit 1
  fi
}
