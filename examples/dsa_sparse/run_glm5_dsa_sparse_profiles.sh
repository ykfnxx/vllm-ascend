#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE_SCRIPT="${SCRIPT_DIR}/serve_glm5_dsa_sparse.sh"
PROFILE_SCRIPT="${SCRIPT_DIR}/profile_glm5_dsa_sparse.py"

MODEL_PATH=""
OUTPUT_ROOT="./dsa-profile-results"
RUN_NAME="dsa-sparse"
SERVED_MODEL_NAME="glm-5"
BATCH_SIZES_CSV="1"
PROMPT_TOKENS=10600
MAX_TOKENS=32
WARMUP_ROUNDS=1
ROUNDS=1
PORT=8077
SERVER_START_TIMEOUT=1800
REQUEST_TIMEOUT=1800
HOT_CPU_BLOCK_MULTIPLE=1
NUM_GPU_BLOCKS_OVERRIDE=128
API_KEY=""
MAX_NUM_SEQS=""
SERVER_PID=""
SERVE_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Run isolated GLM-5 DSA sparse profiles and archive each batch separately.

Usage:
  run_glm5_dsa_sparse_profiles.sh --model-path PATH [options] [-- vllm options]

Required:
  --model-path PATH             GLM-5 model directory

Archive options:
  --output-root PATH            Result root (default: ./dsa-profile-results)
  --run-name NAME               Configuration name (default: dsa-sparse)

Workload options:
  --batch-sizes LIST            Comma-separated concurrency list (default: 1)
  --prompt-tokens N             Minimum prompt tokens (default: 10600)
  --max-tokens N                Output tokens per request (default: 32)
  --warmup-rounds N             Warmup waves per batch (default: 1)
  --rounds N                    Profiled waves per batch (default: 1)
  --max-num-seqs N              Server request capacity; defaults to max batch size

Server options:
  --served-model-name NAME      API model name (default: glm-5)
  --port N                      Local service port (default: 8077)
  --server-start-timeout N      Startup timeout in seconds (default: 1800)
  --request-timeout N           Request timeout in seconds (default: 1800)
  --hot-cpu-block-multiple N    Host hot-cache multiplier (default: 1)
  --num-gpu-blocks-override N   MLA block-pool size (default: 128)
  --api-key KEY                 Bearer token passed to the profile client
  --                            Remaining arguments are passed to vllm serve

Example:
  ./examples/dsa_sparse/run_glm5_dsa_sparse_profiles.sh \
    --model-path /models/GLM-5-W4A8 \
    --output-root /data/profiles \
    --run-name ratio3-hot1 \
    --batch-sizes 1,2,4,8 \
    --num-gpu-blocks-override 768 \
    --max-num-seqs 8 \
    --hot-cpu-block-multiple 1
EOF
}

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "${value}" ]] || fail "${option} requires a value"
}

while (($# > 0)); do
  case "$1" in
    --model-path)
      require_value "$1" "${2:-}"
      MODEL_PATH="$2"
      shift 2
      ;;
    --output-root)
      require_value "$1" "${2:-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-name)
      require_value "$1" "${2:-}"
      RUN_NAME="$2"
      shift 2
      ;;
    --served-model-name)
      require_value "$1" "${2:-}"
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --batch-sizes)
      require_value "$1" "${2:-}"
      BATCH_SIZES_CSV="$2"
      shift 2
      ;;
    --prompt-tokens)
      require_value "$1" "${2:-}"
      PROMPT_TOKENS="$2"
      shift 2
      ;;
    --max-tokens)
      require_value "$1" "${2:-}"
      MAX_TOKENS="$2"
      shift 2
      ;;
    --warmup-rounds)
      require_value "$1" "${2:-}"
      WARMUP_ROUNDS="$2"
      shift 2
      ;;
    --rounds)
      require_value "$1" "${2:-}"
      ROUNDS="$2"
      shift 2
      ;;
    --max-num-seqs)
      require_value "$1" "${2:-}"
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --port)
      require_value "$1" "${2:-}"
      PORT="$2"
      shift 2
      ;;
    --server-start-timeout)
      require_value "$1" "${2:-}"
      SERVER_START_TIMEOUT="$2"
      shift 2
      ;;
    --request-timeout)
      require_value "$1" "${2:-}"
      REQUEST_TIMEOUT="$2"
      shift 2
      ;;
    --hot-cpu-block-multiple)
      require_value "$1" "${2:-}"
      HOT_CPU_BLOCK_MULTIPLE="$2"
      shift 2
      ;;
    --num-gpu-blocks-override)
      require_value "$1" "${2:-}"
      NUM_GPU_BLOCKS_OVERRIDE="$2"
      shift 2
      ;;
    --api-key)
      require_value "$1" "${2:-}"
      API_KEY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      SERVE_EXTRA_ARGS=("$@")
      break
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || fail "--model-path is required"
[[ -d "${MODEL_PATH}" ]] || fail "model directory does not exist: ${MODEL_PATH}"
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  fail "--run-name may contain only letters, digits, dot, underscore, and dash"
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
[[ -x "${SERVE_SCRIPT}" ]] || fail "serve script is not executable: ${SERVE_SCRIPT}"
[[ -f "${PROFILE_SCRIPT}" ]] || fail "profile script is missing: ${PROFILE_SCRIPT}"

IFS=',' read -r -a BATCH_SIZES <<<"${BATCH_SIZES_CSV}"
((${#BATCH_SIZES[@]} > 0)) || fail "--batch-sizes cannot be empty"

MAX_BATCH_SIZE=0
for batch_size in "${BATCH_SIZES[@]}"; do
  [[ "${batch_size}" =~ ^[1-9][0-9]*$ ]] || \
    fail "invalid batch size: ${batch_size}"
  if ((batch_size > MAX_BATCH_SIZE)); then
    MAX_BATCH_SIZE="${batch_size}"
  fi
done

if [[ -z "${MAX_NUM_SEQS}" ]]; then
  MAX_NUM_SEQS="${MAX_BATCH_SIZE}"
fi
[[ "${MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ ]] || \
  fail "--max-num-seqs must be a positive integer"
((MAX_NUM_SEQS >= MAX_BATCH_SIZE)) || \
  fail "--max-num-seqs must be at least the maximum batch size"
[[ "${HOT_CPU_BLOCK_MULTIPLE}" =~ ^[1-9][0-9]*$ ]] || \
  fail "--hot-cpu-block-multiple must be a positive integer"
[[ "${NUM_GPU_BLOCKS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]] || \
  fail "--num-gpu-blocks-override must be a positive integer"

# Each sparse request owns 8192 resident slots, 2048 lookup/query slots, and
# one 128-token tail block. BlockPool also reserves one null block.
DSA_BLOCKS_PER_REQUEST=$(((8192 + 2048 + 128) / 128))
MIN_GPU_BLOCKS=$((MAX_BATCH_SIZE * DSA_BLOCKS_PER_REQUEST + 1))
if ((NUM_GPU_BLOCKS_OVERRIDE < MIN_GPU_BLOCKS)); then
  RECOMMENDED_GPU_BLOCKS=$((((MIN_GPU_BLOCKS + 127) / 128) * 128))
  fail "--num-gpu-blocks-override=${NUM_GPU_BLOCKS_OVERRIDE} is too small for max batch size ${MAX_BATCH_SIZE}; minimum=${MIN_GPU_BLOCKS}, recommended=${RECOMMENDED_GPU_BLOCKS}"
fi

DSA_ADDITIONAL_CONFIG="$(python3 -c \
  'import json,sys; print(json.dumps({"fuse_muls_add": True, "multistream_overlap_shared_expert": True, "dsa_sparse_config": {"enabled": True, "hot_cpu_block_multiple": int(sys.argv[1])}}))' \
  "${HOT_CPU_BLOCK_MULTIPLE}")"

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}/${RUN_STAMP}"
mkdir -p "${RUN_ROOT}"

stop_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

trap stop_server EXIT INT TERM

wait_for_server() {
  local log_path="$1"
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  local health_url="http://127.0.0.1:${PORT}/health"

  while ((SECONDS < deadline)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" 2>/dev/null || true
      SERVER_PID=""
      fail "server exited before becoming ready; inspect ${log_path}"
    fi
    if curl --silent --show-error --fail --max-time 2 \
      "${health_url}" >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done
  fail "server did not become ready in ${SERVER_START_TIMEOUT}s; inspect ${log_path}"
}

write_command() {
  local path="$1"
  shift
  {
    local redact_next=false
    local argument
    for argument in "$@"; do
      if [[ "${redact_next}" == true ]]; then
        printf '%q ' '<redacted>'
        redact_next=false
      elif [[ "${argument}" == "--api-key" ]]; then
        printf '%q ' "${argument}"
        redact_next=true
      elif [[ "${argument}" == --api-key=* ]]; then
        printf '%q ' '--api-key=<redacted>'
      else
        printf '%q ' "${argument}"
      fi
    done
    printf '\n'
  } >"${path}"
}

printf 'Archive root: %s\n' "${RUN_ROOT}"

for batch_size in "${BATCH_SIZES[@]}"; do
  CASE_ROOT="${RUN_ROOT}/bs${batch_size}"
  TRACE_DIR="${CASE_ROOT}/trace"
  SERVER_LOG="${CASE_ROOT}/server.log"
  RESULT_JSON="${CASE_ROOT}/result.json"
  mkdir -p "${TRACE_DIR}"

  PROFILER_CONFIG="$(python3 -c \
    'import json,sys; print(json.dumps({"profiler":"torch","torch_profiler_dir":sys.argv[1]}))' \
    "${TRACE_DIR}")"

  SERVE_COMMAND=(
    "${SERVE_SCRIPT}"
    "${MODEL_PATH}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}"
    --additional-config "${DSA_ADDITIONAL_CONFIG}"
    --profiler-config "${PROFILER_CONFIG}"
    "${SERVE_EXTRA_ARGS[@]}"
  )
  PROFILE_COMMAND=(
    python3 "${PROFILE_SCRIPT}"
    --base-url "http://127.0.0.1:${PORT}"
    --model "${SERVED_MODEL_NAME}"
    --batch-sizes "${batch_size}"
    --prompt-tokens "${PROMPT_TOKENS}"
    --max-tokens "${MAX_TOKENS}"
    --warmup-rounds "${WARMUP_ROUNDS}"
    --rounds "${ROUNDS}"
    --request-timeout "${REQUEST_TIMEOUT}"
    --profile
    --server-log "${SERVER_LOG}"
    --output-json "${RESULT_JSON}"
    --label "${RUN_NAME}-bs${batch_size}"
  )
  if [[ -n "${API_KEY}" ]]; then
    PROFILE_COMMAND+=(--api-key "${API_KEY}")
  fi

  write_command "${CASE_ROOT}/serve-command.txt" "${SERVE_COMMAND[@]}"
  write_command "${CASE_ROOT}/profile-command.txt" "${PROFILE_COMMAND[@]}"

  printf '\n[%s] Starting server; log=%s\n' "bs${batch_size}" "${SERVER_LOG}"
  "${SERVE_COMMAND[@]}" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  wait_for_server "${SERVER_LOG}"

  printf '[%s] Server ready; trace=%s\n' "bs${batch_size}" "${TRACE_DIR}"
  "${PROFILE_COMMAND[@]}"

  stop_server
  printf '[%s] Complete; result=%s\n' "bs${batch_size}" "${RESULT_JSON}"
done

trap - EXIT INT TERM
printf '\nAll profiles completed: %s\n' "${RUN_ROOT}"
