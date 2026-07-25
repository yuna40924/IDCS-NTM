#!/usr/bin/env bash
set -Eeuo pipefail

# One-command Linux environment setup for an NVIDIA A800 server.
# Optional overrides:
#   PYTHON_BIN=python3.11 VENV_DIR=.venv bash scripts/setup_a800.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: this setup script is intended for a Linux A800 server" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi was not found; install/enable the NVIDIA driver first" >&2
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p' | xargs)"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p' | xargs)"
echo "gpu=$GPU_NAME driver=$DRIVER_VERSION"

if [[ "$GPU_NAME" != *A800* ]]; then
  echo "warning: detected '$GPU_NAME' instead of an A800; installation will continue" >&2
fi

version_ge() {
  [[ "$(printf '%s\n' "$2" "$1" | sort -V | sed -n '1p')" == "$2" ]]
}

if version_ge "$DRIVER_VERSION" "560.28.03"; then
  TORCH_VERSION="2.12.1"
  TORCHVISION_VERSION="0.27.1"
  CUDA_WHEEL="cu126"
elif version_ge "$DRIVER_VERSION" "450.80.02"; then
  TORCH_VERSION="2.7.1"
  TORCHVISION_VERSION="0.22.1"
  CUDA_WHEEL="cu118"
  echo "warning: driver < 560.28.03; using the CUDA 11.8 compatibility wheel" >&2
else
  echo "error: NVIDIA driver $DRIVER_VERSION is too old; require >= 450.80.02" >&2
  exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CANDIDATE="$PYTHON_BIN"
else
  PYTHON_CANDIDATE=""
  for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_CANDIDATE="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_CANDIDATE" ]] || ! command -v "$PYTHON_CANDIDATE" >/dev/null 2>&1; then
  echo "error: Python 3.10-3.12 was not found" >&2
  exit 1
fi

"$PYTHON_CANDIDATE" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"error: require Python 3.10-3.12, found {sys.version.split()[0]}"
    )
PY

VENV_PATH="${VENV_DIR:-$PROJECT_ROOT/.venv}"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  "$PYTHON_CANDIDATE" -m venv "$VENV_PATH"
fi

PYTHON="$VENV_PATH/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "https://download.pytorch.org/whl/$CUDA_WHEEL"
"$PYTHON" -m pip install -e '.[dev]'

"$PYTHON" scripts/check_environment.py --require-cuda
"$PYTHON" -m pytest
"$PYTHON" scripts/smoke_test.py --device cuda:0 --num-classes 100

echo
echo "A800 environment is ready."
echo "Activate it with: source '$VENV_PATH/bin/activate'"
echo "Run one paper setting with: bash scripts/run_a800.sh single"

