#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$SCRIPT_DIR/logs/bench_eager_vs_graph_${STAMP}"
mkdir -p "$OUT_DIR"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-10}"
PROMPT_TOKENS="${PROMPT_TOKENS:-131072}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-132000}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-5}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-5}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
bash "$SCRIPT_DIR/install_dmp_dual_attention_runtime.sh"
source "$SCRIPT_DIR/dmp_runtime_env.sh"

{
  echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "pwd: $(pwd)"
  echo "python: $(command -v python3 || command -v python || true)"
  python3 --version 2>&1 || true
  echo "batch_size: $BATCH_SIZE"
  echo "max_tokens: $MAX_TOKENS"
  echo "prompt_tokens: $PROMPT_TOKENS"
  echo "max_model_len: $MAX_MODEL_LEN"
  echo "warmup: $WARMUP"
  echo "repeat: $REPEAT"
  echo "visible_devices: $VISIBLE_DEVICES"
  echo "tensor_parallel_size: $TENSOR_PARALLEL_SIZE"
  echo "dmp_dual_attention: $VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION"
  echo "dmp_stream_mode: $VLLM_ASCEND_DMP_STREAM_MODE"
  echo "dmp_kv_backend: $VLLM_ASCEND_DMP_KV_BACKEND"
  echo "dmp_hixl_config: $VLLM_ASCEND_DMP_HIXL_CONFIG"
  echo "custom_opp_path: $ASCEND_CUSTOM_OPP_PATH"
  echo "vllm_ascend_op_api_path: $VLLM_ASCEND_OP_API_PATH"
} > "$OUT_DIR/RUN_INFO.txt"

run_one() {
  local mode="$1"
  local mode_dir="$OUT_DIR/$mode"
  local log="$mode_dir/full.log"
  local chunks="$mode_dir/chunks"
  mkdir -p "$chunks"

  echo "==== running $mode ===="
  echo "log: $log"

  set +e
  PYTHONUNBUFFERED=1 python3 -u bench_eager_vs_graph.py \
    --mode "$mode" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --prompt-tokens "$PROMPT_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --warmup "$WARMUP" \
    --repeat "$REPEAT" \
    --visible-devices "$VISIBLE_DEVICES" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e

  echo "$status" > "$mode_dir/EXIT_STATUS.txt"
  grep '^BENCH_SUMMARY ' "$log" > "$mode_dir/BENCH_SUMMARY.txt" || true
  grep -nE 'BENCH_SUMMARY|bench_iter|Graph capturing finished|Replaying aclgraph|FULL_DECODE_ONLY|Compilation disabled|Cudagraph is disabled|Traceback|ERROR|Exception|RuntimeError|ValueError|AssertionError' \
    "$log" > "$mode_dir/key.log" || true

  split -l 30 -d -a 3 --additional-suffix=.txt "$mode_dir/key.log" "$chunks/key-"

  wc -l "$log" "$mode_dir/key.log" > "$mode_dir/WC.txt"
  return "$status"
}

status_eager=0
status_graph=0
run_one eager || status_eager=$?
run_one graph || status_graph=$?

{
  echo "Output directory: $OUT_DIR"
  echo "eager_exit: $status_eager"
  echo "graph_exit: $status_graph"
  echo
  echo "Eager summary:"
  cat "$OUT_DIR/eager/BENCH_SUMMARY.txt" 2>/dev/null || true
  echo
  echo "Graph summary:"
  cat "$OUT_DIR/graph/BENCH_SUMMARY.txt" 2>/dev/null || true
  echo
  echo "Key chunks:"
  find "$OUT_DIR" -path '*/chunks/key-*.txt' -print | sort
} | tee "$OUT_DIR/SUMMARY.txt"

if [ "$status_eager" -ne 0 ] || [ "$status_graph" -ne 0 ]; then
  exit 1
fi
