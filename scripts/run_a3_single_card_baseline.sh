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
export BATCH_SIZE="${BATCH_SIZE:-64}"
export PROMPT_TOKENS="${PROMPT_TOKENS:-131072}"
export MAX_TOKENS="${MAX_TOKENS:-10}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-132000}"
export DMP_SCHEME="${DMP_SCHEME:-4}"

export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

source "$SCRIPT_DIR/a3_single_card_runtime_env.sh"
dmp_configure_a3_single_card_rendezvous

runtime_artifacts_ready() {
    local dual_stamp="$SCRIPT_DIR/dmp-runtime/.a3-dual-attention-r4"
    local lookup_stamp="$SCRIPT_DIR/dmp-lookup-maintain/opp/.a3-lookup-maintain-r5"
    local fused_stamp="$SCRIPT_DIR/dmp-fused-indexer-kv-select/opp/.a3-fused-indexer-pool-r1"
    local dual_vendor="$SCRIPT_DIR/dmp-runtime/opp/vendors/customize"
    local lookup_vendor="$SCRIPT_DIR/dmp-lookup-maintain/opp/vendors/customize"

    [[ -f "$dual_stamp" ]] &&
    [[ "$(<"$dual_stamp")" == "A3_DUAL_ATTENTION_RUNTIME_REVISION=4" ]] &&
    [[ -f "$lookup_stamp" ]] &&
    [[ "$(<"$lookup_stamp")" == "A3_LOOKUP_MAINTAIN_RUNTIME_REVISION=5" ]] &&
    [[ -f "$fused_stamp" ]] &&
    [[ "$(<"$fused_stamp")" == "A3_FUSED_INDEXER_POOL_RUNTIME_REVISION=1" ]] &&
    [[ -f "$dual_vendor/op_api/lib/libcust_opapi.so" ]] &&
    [[ -f "$lookup_vendor/op_api/lib/libcust_opapi.so" ]] &&
    [[ -f "$SCRIPT_DIR/dmp-fused-indexer-kv-select/opp/vendors/customize/op_api/lib/libcust_opapi.so" ]] &&
    compgen -G "$SCRIPT_DIR/dmp-runtime/python/custom_ops/*.so" >/dev/null &&
    compgen -G \
        "$SCRIPT_DIR/dmp-lookup-maintain/torch_extension/dmp_lookup_maintain_custom_ops/*.so" \
        >/dev/null &&
    compgen -G \
        "$SCRIPT_DIR/dmp-fused-indexer-kv-select/torch_extension/lightning_indexer_decode_custom_ops/*.so" \
        >/dev/null
}

echo "[bootstrap] Checking and preparing the complete A3 DMP runtime."
if runtime_artifacts_ready; then
    echo "[bootstrap] Reusing the smoke-tested persistent operator runtime."
    DMP_A3_STATIC_CHECK_ONLY=1 \
        bash "$SCRIPT_DIR/check_a3_single_card_prereqs.sh"
    if [[ "${DMP_A3_VALIDATE_RUNTIME:-0}" == "1" ]]; then
        bash "$SCRIPT_DIR/validate_a3_dmp_runtime.sh"
    fi
elif [[ "${DMP_AUTO_BUILD:-1}" == "1" ]]; then
    bash "$SCRIPT_DIR/build_a3_dmp_runtime.sh"
else
    echo "Persistent A3 operator runtime is incomplete." >&2
    echo "Run once with DMP_AUTO_BUILD=1 to build and smoke-test it." >&2
    exit 1
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

echo "[run] Local rendezvous: VLLM_HOST_IP=$VLLM_HOST_IP GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo "[run] Starting single-card scheme $DMP_SCHEME: batch=$BATCH_SIZE prompt=$PROMPT_TOKENS output=$MAX_TOKENS"
exec bash "$SCRIPT_DIR/run_example1_and_split_log.sh"
