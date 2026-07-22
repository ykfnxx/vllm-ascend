#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export SOURCE_MODEL="${SOURCE_MODEL:-/models/GLM-5.1-w4a8}"
export DMP_MIN_MOE_LAYERS="${DMP_MIN_MOE_LAYERS:-1}"
export TENSOR_PARALLEL_SIZE=1
export ENABLE_EXPERT_PARALLEL=0
export MODEL_QUANTIZATION="${MODEL_QUANTIZATION:-ascend}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export PROMPT_TOKENS="${PROMPT_TOKENS:-2048}"
export MAX_TOKENS="${MAX_TOKENS:-10}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-132000}"

export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

echo "[bootstrap] Checking and preparing the complete A3 DMP runtime."
if [[ "${DMP_AUTO_BUILD:-1}" == "1" ]]; then
    bash "$SCRIPT_DIR/build_a3_dmp_runtime.sh"
else
    bash "$SCRIPT_DIR/check_a3_single_card_prereqs.sh"
    bash "$SCRIPT_DIR/validate_a3_dmp_runtime.sh"
fi

layer_args=(
    --config "$SOURCE_MODEL/config.json"
    --minimum-moe-layers "$DMP_MIN_MOE_LAYERS"
)
if [[ -n "${REDUCED_LAYERS:-}" ]]; then
    layer_args+=(--layers "$REDUCED_LAYERS")
fi
export REDUCED_LAYERS="$(
    python3 "$SCRIPT_DIR/select_reduced_layer_count.py" "${layer_args[@]}"
)"
export MODEL_PATH="${MODEL_PATH:-/models-reduced/GLM-5.1-w4a8-${REDUCED_LAYERS}layers-dmp-r2}"

echo "[model] Checking or creating the ${REDUCED_LAYERS}-layer checkpoint."
REDUCED_MODEL_PATH="$MODEL_PATH" \
    bash "$SCRIPT_DIR/prepare_a3_single_card_model.sh"

echo "[run] Starting the single-card scheme-4 baseline."
exec bash "$SCRIPT_DIR/run_example1_and_split_log.sh"
