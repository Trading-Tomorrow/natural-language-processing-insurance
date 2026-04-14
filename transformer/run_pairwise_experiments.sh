#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${SCRIPT_DIR}/data/pairwise_dataset_full.jsonl"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_pairwise.py"
CHECKPOINTS_DIR="${SCRIPT_DIR}/checkpoints"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-mps}"
NUM_EPOCHS="${NUM_EPOCHS:-32}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-balanced}"

run_experiment() {
  local experiment_name="$1"
  shift

  local output_dir="${CHECKPOINTS_DIR}/${experiment_name}"

  if [[ -e "${output_dir}/best_model.pt" && "${FORCE:-0}" != "1" ]]; then
    echo "Skipping ${experiment_name}: ${output_dir}/best_model.pt already exists. Set FORCE=1 to rerun."
    return 0
  fi

  mkdir -p "${output_dir}"

  echo
  echo "=== Running ${experiment_name} ==="
  echo "Output dir: ${output_dir}"

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --dataset-path "${DATASET_PATH}" \
    --class-weighting "${CLASS_WEIGHTING}" \
    --num-epochs "${NUM_EPOCHS}" \
    --device "${DEVICE}" \
    --output-dir "${output_dir}" \
    "$@"
}


run_experiment \
  "pairwise_full_weighted_expanded_ffn512" \
  --batch-size 32 \
  --hidden-size 128 \
  --num-hidden-layers 8 \
  --num-attention-heads 8 \
  --intermediate-size 512

run_experiment \
  "pairwise_full_weighted_expanded_h192_i768" \
  --batch-size 24 \
  --hidden-size 192 \
  --num-hidden-layers 8 \
  --num-attention-heads 8 \
  --intermediate-size 768

run_experiment \
  "pairwise_full_weighted_expanded_h256_i1024" \
  --batch-size 16 \
  --hidden-size 256 \
  --num-hidden-layers 8 \
  --num-attention-heads 8 \
  --intermediate-size 1024

run_experiment \
  "pairwise_full_weighted_expanded_l10_h192_i768" \
  --batch-size 16 \
  --hidden-size 192 \
  --num-hidden-layers 10 \
  --num-attention-heads 8 \
  --intermediate-size 768

echo
echo "All requested experiments finished."
