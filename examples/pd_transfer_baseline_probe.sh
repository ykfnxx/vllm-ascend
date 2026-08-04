#!/usr/bin/env bash
#
# Launch a same-node 1P1D baseline disaggregated prefill test (no DSA Sparse).
#
# This script validates the standard Mooncake control/data path:
#   - two isolated vLLM engines (Prefill TP1 + Decode TP1)
#   - standard KV transfer connector handoff via proxy
#
# It intentionally does not pass any dsa_sparse_config.
#
# Example:
#   bash examples/pd_transfer_baseline_probe.sh \
#     --host-ip 192.168.1.10 \
#     --ifname eth0
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="tiny-random/glm-moe-dsa"
SERVED_MODEL_NAME="glm-moe-dsa"
HOST_IP=""
IFNAME=""
PREFILL_DEVICE="0"
DECODE_DEVICE="1"
PREFILL_HTTP_PORT="18100"
DECODE_HTTP_PORT="18200"
PROXY_HTTP_PORT="18000"
PREFILL_KV_PORT="30000"
DECODE_KV_PORT="30100"
PROMPT_TOKENS="2333"
PROMPT_TOKEN_ID="100"
MAX_TOKENS="1"
MAX_MODEL_LEN="4096"
GPU_MEMORY_UTILIZATION="0.50"
STARTUP_TIMEOUT="900"
LOG_DIR=""

VERIFY_TRANSFER="1"

usage() {
    cat <<'EOF'
Usage:
  pd_transfer_baseline_probe.sh --host-ip IP --ifname NIC [options]

Required:
  --host-ip IP                 Local IP used by Mooncake/HCCL.
  --ifname NIC                 Network interface owning --host-ip.

Options:
  --model MODEL                Model path or Hugging Face ID.
                               Default: tiny-random/glm-moe-dsa
  --served-model-name NAME     OpenAI API model name. Default: glm-moe-dsa
  --prefill-device ID          Prefill physical NPU ID. Default: 0
  --decode-device ID           Decode physical NPU ID. Default: 1
  --prefill-http-port PORT     Prefill HTTP port. Default: 18100
  --decode-http-port PORT      Decode HTTP port. Default: 18200
  --proxy-http-port PORT       Proxy HTTP port. Default: 18000
  --prefill-kv-port PORT       Prefill Mooncake base port. Default: 30000
  --decode-kv-port PORT        Decode Mooncake base port. Default: 30100
  --prompt-tokens N            Repeated input token count. Default: 2333
  --prompt-token-id ID         Repeated vocabulary token ID. Default: 100
  --max-tokens N               Decode token count. Default: 1
  --max-model-len N            Model context limit. Default: 4096
  --gpu-memory-utilization F   Per-engine NPU memory fraction. Default: 0.50
  --startup-timeout SEC        Per-service startup timeout. Default: 900
  --log-dir DIR                Keep logs in DIR. Default: a new /tmp directory
  --no-verify-transfer         Skip log-based transfer assertion.
  -h, --help                   Show this help.
EOF
}

require_value() {
    if (($# < 2)); then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
    fi
}

while (($# > 0)); do
    case "$1" in
        --model)
            require_value "$@"
            MODEL="$2"
            shift 2
            ;;
        --served-model-name)
            require_value "$@"
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        --host-ip)
            require_value "$@"
            HOST_IP="$2"
            shift 2
            ;;
        --ifname)
            require_value "$@"
            IFNAME="$2"
            shift 2
            ;;
        --prefill-device)
            require_value "$@"
            PREFILL_DEVICE="$2"
            shift 2
            ;;
        --decode-device)
            require_value "$@"
            DECODE_DEVICE="$2"
            shift 2
            ;;
        --prefill-http-port)
            require_value "$@"
            PREFILL_HTTP_PORT="$2"
            shift 2
            ;;
        --decode-http-port)
            require_value "$@"
            DECODE_HTTP_PORT="$2"
            shift 2
            ;;
        --proxy-http-port)
            require_value "$@"
            PROXY_HTTP_PORT="$2"
            shift 2
            ;;
        --prefill-kv-port)
            require_value "$@"
            PREFILL_KV_PORT="$2"
            shift 2
            ;;
        --decode-kv-port)
            require_value "$@"
            DECODE_KV_PORT="$2"
            shift 2
            ;;
        --prompt-tokens)
            require_value "$@"
            PROMPT_TOKENS="$2"
            shift 2
            ;;
        --prompt-token-id)
            require_value "$@"
            PROMPT_TOKEN_ID="$2"
            shift 2
            ;;
        --max-tokens)
            require_value "$@"
            MAX_TOKENS="$2"
            shift 2
            ;;
        --max-model-len)
            require_value "$@"
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            require_value "$@"
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --startup-timeout)
            require_value "$@"
            STARTUP_TIMEOUT="$2"
            shift 2
            ;;
        --log-dir)
            require_value "$@"
            LOG_DIR="$2"
            shift 2
            ;;
        --no-verify-transfer)
            VERIFY_TRANSFER="0"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$HOST_IP" || -z "$IFNAME" ]]; then
    echo "--host-ip and --ifname are required." >&2
    usage >&2
    exit 2
fi

for command_name in vllm python3 curl; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command not found: $command_name" >&2
        exit 2
    fi
done

if [[ "$PREFILL_DEVICE" == "$DECODE_DEVICE" ]]; then
    echo "Prefill and Decode must use different physical NPU IDs." >&2
    exit 2
fi

if ((PROMPT_TOKENS + MAX_TOKENS > MAX_MODEL_LEN)); then
    echo "prompt-tokens + max-tokens must not exceed max-model-len." >&2
    exit 2
fi

if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pd-baseline-probe.XXXXXX")"
else
    mkdir -p "$LOG_DIR"
    LOG_DIR="$(cd "$LOG_DIR" && pwd)"
fi

PREFILL_LOG="$LOG_DIR/prefill.log"
DECODE_LOG="$LOG_DIR/decode.log"
PROXY_LOG="$LOG_DIR/proxy.log"
REQUEST_JSON="$LOG_DIR/request.json"
RESPONSE_JSON="$LOG_DIR/response.json"

declare -a CHILD_PIDS=()
PROFILE_STARTED="0"

stop_decode_profile() {
    if [[ "$PROFILE_STARTED" != "1" ]]; then
        return 0
    fi
    # Kept for forward-compatibility if a previous probe profile flow is added.
    if ! curl --fail --silent --show-error \
        --request POST \
        "http://127.0.0.1:$DECODE_HTTP_PORT/stop_profile" \
        >/dev/null; then
        echo "Failed to stop the Decode profiler." >&2
        return 1
    fi
    PROFILE_STARTED="0"
}

cleanup() {
    local pid
    stop_decode_profile || true
    for pid in "${CHILD_PIDS[@]:-}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done
    for pid in "${CHILD_PIDS[@]:-}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done
    echo "Logs kept in: $LOG_DIR"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_health() {
    local service_name="$1"
    local health_url="$2"
    local service_pid="$3"
    local service_log="$4"
    local deadline=$((SECONDS + STARTUP_TIMEOUT))

    while ((SECONDS < deadline)); do
        if curl --fail --silent --show-error "$health_url" >/dev/null 2>&1; then
            echo "$service_name is ready: $health_url"
            return 0
        fi
        if ! kill -0 "$service_pid" >/dev/null 2>&1; then
            echo "$service_name exited before becoming ready." >&2
            tail -n 120 "$service_log" >&2 || true
            return 1
        fi
        sleep 2
    done

    echo "Timed out waiting for $service_name." >&2
    tail -n 120 "$service_log" >&2 || true
    return 1
}

COMMON_NETWORK_ENV=(
    "VLLM_HOST_IP=$HOST_IP"
    "HCCL_IF_IP=$HOST_IP"
    "GLOO_SOCKET_IFNAME=$IFNAME"
    "TP_SOCKET_IFNAME=$IFNAME"
    "HCCL_SOCKET_IFNAME=$IFNAME"
    "OMP_PROC_BIND=false"
    "OMP_NUM_THREADS=1"
)

PREFILL_KV_CONFIG='{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":'"$PREFILL_KV_PORT"',"engine_id":"0","kv_load_failure_policy":"fail","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'
DECODE_KV_CONFIG='{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":'"$DECODE_KV_PORT"',"engine_id":"1","kv_load_failure_policy":"fail","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'

echo "Starting Prefill on physical NPU $PREFILL_DEVICE..."
env \
    "${COMMON_NETWORK_ENV[@]}" \
    "ASCEND_RT_VISIBLE_DEVICES=$PREFILL_DEVICE" \
    vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PREFILL_HTTP_PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 1 \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --kv-transfer-config "$PREFILL_KV_CONFIG" \
    >"$PREFILL_LOG" 2>&1 &
PREFILL_PID=$!
CHILD_PIDS+=("$PREFILL_PID")

echo "Starting Decode on physical NPU $DECODE_DEVICE..."
env \
    "${COMMON_NETWORK_ENV[@]}" \
    "ASCEND_RT_VISIBLE_DEVICES=$DECODE_DEVICE" \
    vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$DECODE_HTTP_PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 1 \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --kv-transfer-config "$DECODE_KV_CONFIG" \
    >"$DECODE_LOG" 2>&1 &
DECODE_PID=$!
CHILD_PIDS+=("$DECODE_PID")

wait_for_health \
    "Prefill" \
    "http://127.0.0.1:$PREFILL_HTTP_PORT/health" \
    "$PREFILL_PID" \
    "$PREFILL_LOG"
wait_for_health \
    "Decode" \
    "http://127.0.0.1:$DECODE_HTTP_PORT/health" \
    "$DECODE_PID" \
    "$DECODE_LOG"

echo "Starting the standard P/D proxy..."
python3 "$SCRIPT_DIR/disaggregated_prefill_v1/load_balance_proxy_server_example.py" \
    --host 127.0.0.1 \
    --port "$PROXY_HTTP_PORT" \
    --prefiller-hosts 127.0.0.1 \
    --prefiller-ports "$PREFILL_HTTP_PORT" \
    --decoder-hosts 127.0.0.1 \
    --decoder-ports "$DECODE_HTTP_PORT" \
    --log-level DEBUG \
    >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
CHILD_PIDS+=("$PROXY_PID")

wait_for_health \
    "P/D proxy" \
    "http://127.0.0.1:$PROXY_HTTP_PORT/healthcheck" \
    "$PROXY_PID" \
    "$PROXY_LOG"

python3 - \
    "$REQUEST_JSON" \
    "$SERVED_MODEL_NAME" \
    "$PROMPT_TOKENS" \
    "$PROMPT_TOKEN_ID" \
    "$MAX_TOKENS" <<'PY'
import json
import sys

output_path, model_name, prompt_tokens, token_id, max_tokens = sys.argv[1:]
payload = {
    "model": model_name,
    "prompt": [int(token_id)] * int(prompt_tokens),
    "max_tokens": int(max_tokens),
    "temperature": 0.0,
    "stream": False,
}
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(payload, output_file)
PY

echo "Submitting a $PROMPT_TOKENS-token request through the P/D proxy..."
HTTP_CODE="$(
    curl --silent --show-error \
        --output "$RESPONSE_JSON" \
        --write-out '%{http_code}' \
        --header 'Content-Type: application/json' \
        --request POST \
        --data-binary "@$REQUEST_JSON" \
        "http://127.0.0.1:$PROXY_HTTP_PORT/v1/completions"
)"

if [[ "$HTTP_CODE" != 2* ]]; then
    echo "P/D request failed with HTTP $HTTP_CODE." >&2
    python3 -m json.tool "$RESPONSE_JSON" >&2 2>/dev/null || cat "$RESPONSE_JSON" >&2
    echo "Last Prefill log lines:" >&2
    tail -n 120 "$PREFILL_LOG" >&2 || true
    echo "Last Decode log lines:" >&2
    tail -n 120 "$DECODE_LOG" >&2 || true
    echo "Last proxy log lines:" >&2
    tail -n 80 "$PROXY_LOG" >&2 || true
    exit 1
fi

echo "P/D request completed with HTTP $HTTP_CODE."
python3 -m json.tool "$RESPONSE_JSON" 2>/dev/null || cat "$RESPONSE_JSON"

if [[ "$VERIFY_TRANSFER" == "1" ]]; then
    if ! grep -Eq 'Mooncake transfer request|Mooncake kv transfer meta|Mooncake transfer failed' "$PREFILL_LOG" "$DECODE_LOG"; then
        echo "Did not find any Mooncake transfer log in P/D logs. Raw check lines:" >&2
        tail -n 120 "$PREFILL_LOG" >&2 || true
        tail -n 120 "$DECODE_LOG" >&2 || true
        exit 1
    fi
    if grep -Eq 'Mooncake transfer failed for request|Failed to transfer KV cache' "$PREFILL_LOG" "$DECODE_LOG"; then
        echo "Mooncake transfer reported failure in logs." >&2
        grep -En 'Mooncake transfer failed|Failed to transfer KV cache' "$DECODE_LOG" >&2 || true
        exit 1
    fi
fi

echo
echo "PASS: Baseline disaggregated prefill/decode flow completed."
echo "If VERIFY_TRANSFER=1, logs show a valid Mooncake transfer trace:"
echo "  grep -En 'Mooncake register kv caches metadata|Mooncake kv transfer meta|Mooncake transfer request|Mooncake transfer failed|MooncakeConnector' '$DECODE_LOG' '$PREFILL_LOG'"
echo "Proxy log:"
echo "  tail -n 80 '$PROXY_LOG'"
echo "Response:"
echo "  cat '$RESPONSE_JSON'"
echo "Logs kept in: $LOG_DIR"
