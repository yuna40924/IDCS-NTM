#!/usr/bin/env bash
set -Eeuo pipefail

# Direct A800 launcher for Section 6.3.1.
#
# Examples:
#   bash scripts/run_a800.sh single
#   bash scripts/run_a800.sh single cifar100 asymmetric_i 0.4 1
#   bash scripts/run_a800.sh full
#   bash scripts/run_a800.sh dry-run
#
# Optional environment variables:
#   GPU_ID=0 DATA_ROOT=/data/cifar OUTPUT_ROOT=/results/idcs_ntm
#   NUM_WORKERS=8 OVERWRITE=1 DOWNLOAD=0

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-single}"
VENV_PATH="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON="$VENV_PATH/bin/python"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/section_6_3_1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OVERWRITE="${OVERWRITE:-0}"
DOWNLOAD="${DOWNLOAD:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: environment not found at $VENV_PATH" >&2
  echo "run: bash scripts/setup_a800.sh" >&2
  exit 1
fi

if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "error: GPU_ID must be one non-negative integer" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

COMMON=(
  --device cuda:0
  --data-root "$DATA_ROOT"
  --output-root "$OUTPUT_ROOT"
  --num-workers "$NUM_WORKERS"
)
if [[ "$DOWNLOAD" == "1" ]]; then
  COMMON+=(--download)
else
  COMMON+=(--no-download)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  COMMON+=(--overwrite)
fi

"$PYTHON" scripts/check_environment.py --require-cuda

run_single() {
  local dataset="${2:-cifar10}"
  local noise_type="${3:-symmetric}"
  local noise_rate="${4:-0.4}"
  local seed="${5:-1}"
  local rate_name
  rate_name="$("$PYTHON" -c 'import sys; print(f"{float(sys.argv[1]):.2f}".replace(".", "p"))' "$noise_rate")"
  local base="$OUTPUT_ROOT/$dataset/${noise_type}_${rate_name}/seed_${seed}"
  local ce_checkpoint="$base/ce/checkpoint_last.pt"

  echo "running dataset=$dataset noise=$noise_type rate=$noise_rate seed=$seed gpu=$GPU_ID"
  for method in ce forward idcs_ntm; do
    local summary="$base/$method/summary.json"
    if [[ -f "$summary" && "$OVERWRITE" != "1" ]]; then
      echo "skip completed method=$method ($summary)"
      continue
    fi
    local command=(
      "$PYTHON" -m idcs_ntm.cli
      --dataset "$dataset"
      --noise-type "$noise_type"
      --noise-rate "$noise_rate"
      --method "$method"
      --seed "$seed"
      "${COMMON[@]}"
    )
    if [[ "$method" != "ce" ]]; then
      if [[ ! -f "$ce_checkpoint" ]]; then
        echo "error: CE checkpoint is missing: $ce_checkpoint" >&2
        echo "rerun CE with OVERWRITE=1 if its previous run was interrupted" >&2
        exit 1
      fi
      command+=(--ce-checkpoint "$ce_checkpoint")
    fi
    if [[ -f "$base/$method/metrics.jsonl" && "$OVERWRITE" != "1" ]]; then
      echo "restart incomplete method=$method"
      command+=(--overwrite)
    fi
    "${command[@]}"
  done
}

case "$MODE" in
  single)
    run_single "$@"
    ;;
  full|dry-run)
    SWEEP=(
      "$PYTHON" -m idcs_ntm.sweep
      --config configs/section_6_3_1.yaml
      --device cuda:0
      --data-root "$DATA_ROOT"
      --output-root "$OUTPUT_ROOT"
      --num-workers "$NUM_WORKERS"
    )
    if [[ "$DOWNLOAD" == "1" ]]; then
      SWEEP+=(--download)
    else
      SWEEP+=(--no-download)
    fi
    if [[ "$MODE" == "dry-run" ]]; then
      SWEEP+=(--dry-run)
    fi
    if [[ "$OVERWRITE" == "1" ]]; then
      SWEEP+=(--overwrite)
    fi
    "${SWEEP[@]}"
    ;;
  *)
    echo "usage: bash scripts/run_a800.sh {single|full|dry-run} [dataset noise_type rate seed]" >&2
    exit 2
    ;;
esac
