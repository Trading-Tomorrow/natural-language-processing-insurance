#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-mps}"
NUM_EPOCHS="${NUM_EPOCHS:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FORCE="${FORCE:-0}"

PLAIN_DATASET_PATH="${SCRIPT_DIR}/data/pairwise_dataset_full.jsonl"
CONTEXTUAL_DATASET_PATH="${SCRIPT_DIR}/data/pairwise_dataset_full_contextual.jsonl"

run_training() {
  local output_dir="$1"
  shift

  if [[ "${FORCE}" != "1" && -f "${output_dir}/best_model.pt" ]]; then
    echo "Skipping existing run: ${output_dir}"
    return
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_pairwise.py" \
    --dataset-path "$@" \
    --class-weighting balanced \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-size 128 \
    --num-hidden-layers 8 \
    --num-attention-heads 8 \
    --intermediate-size 512 \
    --device "${DEVICE}" \
    --output-dir "${output_dir}"
}

echo "Building plain pairwise dataset..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_pairwise_dataset.py" --context-mode plain

echo "Building contextual pairwise dataset..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_pairwise_dataset.py" --context-mode contextual

echo "Running ablation: comparison head on plain input..."
run_training \
  "${SCRIPT_DIR}/checkpoints/pairwise_full_weighted_expanded_ffn512_plain_comparison_head" \
  "${PLAIN_DATASET_PATH}"

echo "Running ablation: no comparison head on contextual input..."
run_training \
  "${SCRIPT_DIR}/checkpoints/pairwise_full_weighted_expanded_ffn512_context_no_comparison_head" \
  "${CONTEXTUAL_DATASET_PATH}" \
  --disable-pairwise-comparison-head

