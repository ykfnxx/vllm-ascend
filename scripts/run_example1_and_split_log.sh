#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$SCRIPT_DIR/logs/example1_${STAMP}"
CHUNK_DIR="$OUT_DIR/chunks"
LOG="$OUT_DIR/example1.full.log"

export BATCH_SIZE="${BATCH_SIZE:-64}"
export MAX_TOKENS="${MAX_TOKENS:-10}"
export PROMPT_TOKENS="${PROMPT_TOKENS:-131072}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-132000}"
export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"
export DMP_SCHEME="${DMP_SCHEME:-4}"
source "$SCRIPT_DIR/a3_single_card_runtime_env.sh"
dmp_configure_a3_single_card_rendezvous
case "$DMP_SCHEME" in
  1)
    export VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN=0
    export VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT=0
    export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION=0
    ;;
  2)
    export VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN=0
    export VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT=0
    export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION=1
    source "$SCRIPT_DIR/dmp_runtime_env.sh"
    ;;
  3)
    export VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN=0
    export VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT=1
    export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION=0
    source "$SCRIPT_DIR/dmp_fused_indexer_runtime_env.sh"
    ;;
  4)
    export VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN=1
    export VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT=0
    export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION=0
    source "$SCRIPT_DIR/dmp_lookup_maintain_runtime_env.sh"
    ;;
  *)
    echo "DMP_SCHEME must be one of 1, 2, 3, or 4; got: $DMP_SCHEME" >&2
    exit 2
    ;;
esac
dmp_activate_a3_model_runtime

mkdir -p "$CHUNK_DIR"

{
  echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "pwd: $(pwd)"
  echo "python: $(command -v python3 || command -v python || true)"
  python3 --version 2>&1 || true
  echo "script: $SCRIPT_DIR/example1.py"
  echo "batch_size: $BATCH_SIZE"
  echo "max_tokens: $MAX_TOKENS"
  echo "prompt_tokens: $PROMPT_TOKENS"
  echo "max_model_len: $MAX_MODEL_LEN"
  echo "visible_devices: $VISIBLE_DEVICES"
  echo "tensor_parallel_size: $TENSOR_PARALLEL_SIZE"
  echo "vllm_host_ip: ${VLLM_HOST_IP:-<auto>}"
  echo "gloo_socket_ifname: ${GLOO_SOCKET_IFNAME:-<auto>}"
  echo "model_path: ${MODEL_PATH:-/models/GLM-5.1-w4a8}"
  echo "reduced_layers: ${REDUCED_LAYERS:-<not-set>}"
  echo "model_quantization: ${MODEL_QUANTIZATION:-ascend}"
  echo "transformers_runtime: ${DMP_MODEL_RUNTIME_INFO:-<unknown>}"
  echo "enable_expert_parallel: ${ENABLE_EXPERT_PARALLEL:-0}"
  echo "dmp_scheme: $DMP_SCHEME"
  echo "dmp_lookup_maintain: $VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN"
  echo "dmp_fused_indexer_kv_select: $VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT"
  echo "dmp_dual_attention: $VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION"
  echo "dmp_stream_mode: ${VLLM_ASCEND_DMP_STREAM_MODE:-<unused>}"
  echo "dmp_kv_backend: ${VLLM_ASCEND_DMP_KV_BACKEND:-<unused>}"
  echo "dmp_hixl_config: ${VLLM_ASCEND_DMP_HIXL_CONFIG:-<unused>}"
  echo "custom_opp_path: ${ASCEND_CUSTOM_OPP_PATH:-<unset>}"
  echo "vllm_ascend_op_api_path: ${VLLM_ASCEND_OP_API_PATH:-<auto>}"
} > "$OUT_DIR/RUN_INFO.txt"

echo "Running example1.py ..."
echo "Full log: $LOG"
echo

set +e
PYTHONUNBUFFERED=1 python3 -u example1.py 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

echo "$status" > "$OUT_DIR/EXIT_STATUS.txt"

python3 "$SCRIPT_DIR/summarize_distributed_init.py" "$LOG" \
  | tee "$OUT_DIR/DISTRIBUTED_INIT_TIMING.txt"

split -l 30 -d -a 3 --additional-suffix=.txt "$LOG" "$CHUNK_DIR/part-"

wc -l "$LOG" > "$OUT_DIR/WC.txt"
sha256sum "$LOG" "$CHUNK_DIR"/part-*.txt > "$OUT_DIR/SHA256SUMS"

{
  echo "Output directory: $OUT_DIR"
  echo "Exit status: $status"
  echo
  cat "$OUT_DIR/WC.txt"
  echo
  echo "Chunks:"
  ls "$CHUNK_DIR"/part-*.txt
} | tee "$OUT_DIR/SUMMARY.txt"

exit "$status"
