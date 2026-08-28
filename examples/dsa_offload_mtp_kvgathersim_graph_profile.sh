#!/usr/bin/env bash
# Profile the A5 Decode graph path with MTP, batch lookup, and ASU KV Gather.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CUSTOM_OP_VENDOR_PATH="$REPO_ROOT/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
CUSTOM_OP_API_DIR="$CUSTOM_OP_VENDOR_PATH/op_api/lib"
CUSTOM_OP_API_LIB="$CUSTOM_OP_API_DIR/libcust_opapi.so"
CUSTOM_OP_KERNEL_ROOT="$CUSTOM_OP_VENDOR_PATH/op_impl/ai_core/tbe/kernel"
CUSTOM_OP_INDEX_TOOL="$REPO_ROOT/scripts/a5_kvgather_kernel_index.py"

MODEL="${MODEL_PATH_IN_CONTAINER:-/models/glm-moe-dsa}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-moe-dsa}"
HOST_IP="${A5_PD_HOST_IP:-90.90.93.29}"
IFNAME="${A5_PD_IFNAME:-ens6f1}"
PREFILL_DEVICE="${A5_PREFILL_DEVICE:-3}"
DECODE_DEVICE="${A5_DECODE_DEVICE:-5}"
PREFILL_HTTP_PORT="${A5_PREFILL_HTTP_PORT:-18100}"
DECODE_HTTP_PORT="${A5_DECODE_HTTP_PORT:-18200}"
PROXY_HTTP_PORT="${A5_PROXY_HTTP_PORT:-18000}"
PREFILL_KV_PORT="${A5_PREFILL_KV_PORT:-30000}"
DECODE_KV_PORT="${A5_DECODE_KV_PORT:-30100}"
BATCH_SIZE=64
PROMPT_TOKENS=131072
PROMPT_TOKEN_ID="${A5_PROMPT_TOKEN_ID:-100}"
MAX_TOKENS=10
MAX_MODEL_LEN=131200
BLOCK_SIZE=128
LOCAL_SHM_DIR="${A5_LOCAL_SHM_DIR:-/dev/shm/vllm-ascend-local-kv}"
LOCAL_SHM_NAMESPACE="${A5_LOCAL_SHM_NAMESPACE:-dsa-offload-kvgather-graph}"
LOCAL_SHM_TIMEOUT="${A5_LOCAL_SHM_TIMEOUT:-120}"
PREFILL_CHUNK_SIZE="${A5_PREFILL_CHUNK_SIZE:-4096}"
GPU_MEMORY_UTILIZATION="${A5_GPU_MEMORY_UTILIZATION:-0.90}"
STARTUP_TIMEOUT="${A5_STARTUP_TIMEOUT:-1800}"
REQUEST_TIMEOUT="${A5_REQUEST_TIMEOUT:-7200}"
LOG_DIR="${A5_SERVICE_LOG_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/a5-kvgather-sim.XXXXXX")}"
PROFILE_DIR="${A5_PROFILE_DIR:-$LOG_DIR/profile}"

[[ "$PREFILL_DEVICE" != "$DECODE_DEVICE" ]] || {
    echo "Prefill and Decode must use different physical NPUs." >&2
    exit 2
}
[[ "$BATCH_SIZE" == "64" && "$PROMPT_TOKENS" == "131072" && \
   "$MAX_TOKENS" == "10" ]] || {
    echo "This validation entry is fixed to batch64 x prompt128K x output10." >&2
    exit 2
}
((PROMPT_TOKENS + MAX_TOKENS <= MAX_MODEL_LEN)) || {
    echo "prompt-tokens + max-tokens exceeds max-model-len." >&2
    exit 2
}
for command_name in vllm python3 curl; do
    command -v "$command_name" >/dev/null || {
        echo "Required command not found: $command_name" >&2
        exit 2
    }
done
[[ -r "$CUSTOM_OP_API_LIB" ]] || {
    echo "Custom op API library is missing: $CUSTOM_OP_API_LIB" >&2
    exit 2
}
if [[ ":${ASCEND_CUSTOM_OPP_PATH:-}:" != *":$CUSTOM_OP_VENDOR_PATH:"* ]]; then
    export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OP_VENDOR_PATH${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
fi
if [[ ":${LD_LIBRARY_PATH:-}:" != *":$CUSTOM_OP_API_DIR:"* ]]; then
    export LD_LIBRARY_PATH="$CUSTOM_OP_API_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
echo "A5_CUSTOM_OP_API_READY: path=$CUSTOM_OP_API_LIB"
python3 "$CUSTOM_OP_INDEX_TOOL" --kernel-root "$CUSTOM_OP_KERNEL_ROOT"

mkdir -p "$LOG_DIR" "$PROFILE_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
PROFILE_DIR="$(cd "$PROFILE_DIR" && pwd)"
if find "$PROFILE_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Profile directory must be empty: $PROFILE_DIR" >&2
    exit 2
fi

PREFILL_LOG="$LOG_DIR/prefill.log"
DECODE_LOG="$LOG_DIR/decode.log"
PROXY_LOG="$LOG_DIR/proxy.log"
REQUEST_JSON="$LOG_DIR/request.json"
RESPONSE_JSON="$LOG_DIR/response.json"
RESULT_JSON="$LOG_DIR/validation.json"
PROFILE_STARTED=0
declare -a CHILD_PIDS=()

stop_decode_profile() {
    if [[ "$PROFILE_STARTED" == "1" ]]; then
        curl --fail --silent --show-error --request POST \
            "http://127.0.0.1:$DECODE_HTTP_PORT/stop_profile" >/dev/null
        PROFILE_STARTED=0
    fi
}

cleanup() {
    local pid
    stop_decode_profile || true
    for pid in "${CHILD_PIDS[@]:-}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
    for pid in "${CHILD_PIDS[@]:-}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_health() {
    local name="$1" url="$2" pid="$3" log="$4"
    local deadline=$((SECONDS + STARTUP_TIMEOUT))
    while ((SECONDS < deadline)); do
        if curl --fail --silent "$url" >/dev/null 2>&1; then
            echo "$name is ready: $url"
            return 0
        fi
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            echo "$name exited before becoming ready." >&2
            tail -n 160 "$log" >&2 || true
            return 1
        fi
        if grep -qE \
            'EngineCore failed to start|EngineDeadError|Engine core initialization failed' \
            "$log" 2>/dev/null; then
            echo "$name EngineCore failed before readiness." >&2
            tail -n 160 "$log" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "Timed out waiting for $name." >&2
    tail -n 160 "$log" >&2 || true
    return 1
}

COMMON_ENV=(
    "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
    "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    "VLLM_HOST_IP=$HOST_IP"
    "HCCL_IF_IP=$HOST_IP"
    "GLOO_SOCKET_IFNAME=$IFNAME"
    "TP_SOCKET_IFNAME=$IFNAME"
    "HCCL_SOCKET_IFNAME=$IFNAME"
    "OMP_PROC_BIND=false"
    "OMP_NUM_THREADS=1"
    "VLLM_ASCEND_ENABLE_DMP=0"
    # The native SFA preprocessing path is used for this graph profile.
    "VLLM_ASCEND_ENABLE_MLAPO=0"
)

PROFILER_CONFIG="$(python3 - "$PROFILE_DIR" <<'PY'
import json
import sys

print(json.dumps({
    "profiler": "torch",
    "torch_profiler_dir": sys.argv[1],
    "torch_profiler_with_stack": False,
}))
PY
)"
PREFILL_DSA_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":false},"dsa_offload":{"io_backend":"mock"}}'
DECODE_DSA_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":false},"dsa_offload":{"io_backend":"kvgather_sim"}}'
PREFILL_SPEC_CONFIG='{"method":"deepseek_mtp","num_speculative_tokens":1}'
DECODE_SPEC_CONFIG='{"method":"deepseek_mtp","num_speculative_tokens":1}'
DECODE_GRAPH_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[64,128],"max_cudagraph_capture_size":128}'
PREFILL_KV_CONFIG="{\"kv_connector\":\"LocalShmConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":$PREFILL_KV_PORT,\"engine_id\":\"dsa-offload-prefill\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":1},\"decode\":{\"dp_size\":1,\"tp_size\":1},\"shm_dir\":\"$LOCAL_SHM_DIR\",\"shm_namespace\":\"$LOCAL_SHM_NAMESPACE\",\"shm_timeout\":$LOCAL_SHM_TIMEOUT}}"
DECODE_KV_CONFIG="{\"kv_connector\":\"LocalShmConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":$DECODE_KV_PORT,\"engine_id\":\"dsa-offload-decode\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":1},\"decode\":{\"dp_size\":1,\"tp_size\":1},\"shm_dir\":\"$LOCAL_SHM_DIR\",\"shm_namespace\":\"$LOCAL_SHM_NAMESPACE\",\"shm_timeout\":$LOCAL_SHM_TIMEOUT}}"

echo "Starting eager MTP Prefill on physical NPU $PREFILL_DEVICE..."
env "${COMMON_ENV[@]}" \
    "ASCEND_RT_VISIBLE_DEVICES=$PREFILL_DEVICE" \
    vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$PREFILL_HTTP_PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$BATCH_SIZE" \
    --max-num-batched-tokens "$PREFILL_CHUNK_SIZE" \
    --block-size "$BLOCK_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code --enforce-eager --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --additional-config "$PREFILL_DSA_CONFIG" \
    --speculative-config "$PREFILL_SPEC_CONFIG" \
    --kv-transfer-config "$PREFILL_KV_CONFIG" \
    >"$PREFILL_LOG" 2>&1 &
PREFILL_PID=$!
CHILD_PIDS+=("$PREFILL_PID")

echo "Starting MTP kvgather_sim Decode Graph on physical NPU $DECODE_DEVICE..."
env "${COMMON_ENV[@]}" \
    "ASCEND_RT_VISIBLE_DEVICES=$DECODE_DEVICE" \
    vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$DECODE_HTTP_PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$BATCH_SIZE" \
    --max-num-batched-tokens 128 \
    --block-size "$BLOCK_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code --no-enable-prefix-caching \
    --compilation-config "$DECODE_GRAPH_CONFIG" \
    --additional-config "$DECODE_DSA_CONFIG" \
    --speculative-config "$DECODE_SPEC_CONFIG" \
    --kv-transfer-config "$DECODE_KV_CONFIG" \
    --profiler-config "$PROFILER_CONFIG" \
    >"$DECODE_LOG" 2>&1 &
DECODE_PID=$!
CHILD_PIDS+=("$DECODE_PID")

wait_for_health Prefill \
    "http://127.0.0.1:$PREFILL_HTTP_PORT/health" \
    "$PREFILL_PID" "$PREFILL_LOG"
wait_for_health Decode \
    "http://127.0.0.1:$DECODE_HTTP_PORT/health" \
    "$DECODE_PID" "$DECODE_LOG"

echo "Starting the standard P/D proxy..."
python3 "$SCRIPT_DIR/disaggregated_prefill_v1/load_balance_proxy_server_example.py" \
    --host 127.0.0.1 --port "$PROXY_HTTP_PORT" \
    --prefiller-hosts 127.0.0.1 --prefiller-ports "$PREFILL_HTTP_PORT" \
    --decoder-hosts 127.0.0.1 --decoder-ports "$DECODE_HTTP_PORT" \
    --log-level INFO >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
CHILD_PIDS+=("$PROXY_PID")
wait_for_health "P/D proxy" \
    "http://127.0.0.1:$PROXY_HTTP_PORT/healthcheck" \
    "$PROXY_PID" "$PROXY_LOG"

python3 - "$REQUEST_JSON" "$SERVED_MODEL_NAME" "$BATCH_SIZE" \
    "$PROMPT_TOKENS" "$PROMPT_TOKEN_ID" "$MAX_TOKENS" <<'PY'
import json
import sys

path, model, batch, prompt_tokens, token_id, max_tokens = sys.argv[1:]
prompt = [int(token_id)] * int(prompt_tokens)
payload = {
    "model": model,
    "prompt": [prompt] * int(batch),
    "max_tokens": int(max_tokens),
    "temperature": 0.0,
    "ignore_eos": True,
    "stream": False,
}
with open(path, "w", encoding="utf-8") as output:
    json.dump(payload, output, separators=(",", ":"))
PY

echo "Starting Decode-only profiling."
curl --fail --silent --show-error --request POST \
    "http://127.0.0.1:$DECODE_HTTP_PORT/start_profile" >/dev/null
PROFILE_STARTED=1

echo "Submitting batch=64 prompt=131072 output=10 through P/D."
HTTP_CODE="$(curl --silent --show-error --max-time "$REQUEST_TIMEOUT" \
    --output "$RESPONSE_JSON" --write-out '%{http_code}' \
    --header 'Content-Type: application/json' --request POST \
    --data-binary "@$REQUEST_JSON" \
    "http://127.0.0.1:$PROXY_HTTP_PORT/v1/completions")"
stop_decode_profile

if [[ "$HTTP_CODE" != 2* ]]; then
    echo "P/D request failed with HTTP $HTTP_CODE." >&2
    tail -n 200 "$DECODE_LOG" >&2 || true
    exit 1
fi

echo "Analyzing Decode-only profile..."
python3 - "$PROFILE_DIR" <<'PY'
import sys
import time
from pathlib import Path

from torch_npu.profiler.profiler import analyse

root = Path(sys.argv[1])
deadline = time.monotonic() + 180
traces = []
while time.monotonic() < deadline:
    traces = sorted(path for path in root.rglob("*_ascend_pt") if path.is_dir())
    if traces:
        break
    time.sleep(1)
if not traces:
    raise SystemExit(f"No Ascend profiler trace found under {root}")
for trace in traces:
    analyse(str(trace))
PY

python3 "$SCRIPT_DIR/dsa_offload_mtp_kvgathersim_graph_validate.py" \
    --decode-log "$DECODE_LOG" \
    --response-json "$RESPONSE_JSON" \
    --profile-dir "$PROFILE_DIR" \
    --batch-size "$BATCH_SIZE" \
    --prompt-tokens "$PROMPT_TOKENS" \
    --output-tokens "$MAX_TOKENS" >"$RESULT_JSON"
cat "$RESULT_JSON"
echo "A5_DSA_OFFLOAD_MTP_KVGATHER_SIM_GRAPH_PROFILE_PASSED"
echo "logs=$LOG_DIR"
echo "profile=$PROFILE_DIR"
