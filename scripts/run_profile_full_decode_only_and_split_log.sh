#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${MODE:-graph}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-10}"
PROMPT_TOKENS="${PROMPT_TOKENS:-131072}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-132000}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-5}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
bash "$SCRIPT_DIR/install_dmp_dual_attention_runtime.sh"
source "$SCRIPT_DIR/dmp_runtime_env.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"

OUT_DIR="$SCRIPT_DIR/logs/profile_full_decode_only_${MODE}_${STAMP}"
PROFILE_DIR="$OUT_DIR/profile"
CHUNK_DIR="$OUT_DIR/chunks"
FULL_LOG="$OUT_DIR/full.log"
ANALYSE_LOG="$OUT_DIR/analyse.log"
KEY_LOG="$OUT_DIR/key.log"

mkdir -p "$PROFILE_DIR" "$CHUNK_DIR"

{
  echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "pwd: $(pwd)"
  echo "mode: $MODE"
  echo "batch_size: $BATCH_SIZE"
  echo "max_tokens: $MAX_TOKENS"
  echo "prompt_tokens: $PROMPT_TOKENS"
  echo "max_model_len: $MAX_MODEL_LEN"
  echo "visible_devices: $VISIBLE_DEVICES"
  echo "tensor_parallel_size: $TENSOR_PARALLEL_SIZE"
  echo "dmp_dual_attention: $VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION"
  echo "dmp_stream_mode: $VLLM_ASCEND_DMP_STREAM_MODE"
  echo "dmp_kv_backend: $VLLM_ASCEND_DMP_KV_BACKEND"
  echo "dmp_hixl_config: $VLLM_ASCEND_DMP_HIXL_CONFIG"
  echo "custom_opp_path: $ASCEND_CUSTOM_OPP_PATH"
  echo "vllm_ascend_op_api_path: $VLLM_ASCEND_OP_API_PATH"
  echo "profile_dir: $PROFILE_DIR"
  echo "python: $(command -v python3 || command -v python || true)"
  python3 --version 2>&1 || true
} > "$OUT_DIR/RUN_INFO.txt"

echo "==== running profile mode=$MODE ===="
echo "full log: $FULL_LOG"
echo "profile dir: $PROFILE_DIR"

set +e
PYTHONUNBUFFERED=1 python3 -u profile_full_decode_only.py \
  --mode "$MODE" \
  --batch-size "$BATCH_SIZE" \
  --max-tokens "$MAX_TOKENS" \
  --prompt-tokens "$PROMPT_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --visible-devices "$VISIBLE_DEVICES" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --profile-dir "$PROFILE_DIR" \
  2>&1 | tee "$FULL_LOG"
run_status=${PIPESTATUS[0]}
set -e
echo "$run_status" > "$OUT_DIR/EXIT_STATUS.txt"

echo "==== analysing profile output ===="
set +e
PYTHONUNBUFFERED=1 python3 -u analyse_npu_profile.py "$PROFILE_DIR" \
  2>&1 | tee "$ANALYSE_LOG"
analyse_status=${PIPESTATUS[0]}
set -e
echo "$analyse_status" > "$OUT_DIR/ANALYSE_EXIT_STATUS.txt"

grep -nE 'profile_|prefix_cache_|Graph capturing finished|Replaying aclgraph|FULL_DECODE_ONLY|Compilation disabled|Cudagraph is disabled|Incorrect schedule|profiling data cannot be parsed|analyse|Traceback|ERROR|Exception|RuntimeError|ValueError|AssertionError|Processed prompts' \
  "$FULL_LOG" "$ANALYSE_LOG" > "$KEY_LOG" || true

split -l 30 -d -a 3 --additional-suffix=.txt "$KEY_LOG" "$CHUNK_DIR/key-"

{
  echo "Output directory: $OUT_DIR"
  echo "Run exit status: $run_status"
  echo "Analyse exit status: $analyse_status"
  echo
  echo "RUN_INFO:"
  cat "$OUT_DIR/RUN_INFO.txt"
  echo
  echo "Profile tree:"
  find "$PROFILE_DIR" -maxdepth 4 -print | sort | head -120
  echo
  echo "Key chunks:"
  find "$CHUNK_DIR" -type f -name 'key-*.txt' -print | sort
} | tee "$OUT_DIR/SUMMARY.txt"

if [ "$run_status" -ne 0 ]; then
  exit "$run_status"
fi

# A profile parse failure should not hide a successful model run, because the
# raw profile tree is still useful evidence. Keep the script exit tied to model
# execution and report analyse status in SUMMARY.txt.
exit 0
