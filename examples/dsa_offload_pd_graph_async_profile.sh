#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#
# Profile the DSA Offload P/D path with Decode Graph and async scheduling.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  dsa_offload_pd_graph_async_profile.sh \
    MODEL PREFILL_DEVICE DECODE_DEVICE INPUT_LENGTH OUTPUT_LENGTH \
    IO_BACKEND BATCH_SIZE

Arguments:
  MODEL           Local model path or Hugging Face ID.
  PREFILL_DEVICE  Physical NPU for the eager Prefill service.
  DECODE_DEVICE   Physical NPU for the Graph Decode service.
  INPUT_LENGTH    Tokens per prompt.
  OUTPUT_LENGTH   Generated tokens per prompt.
  IO_BACKEND      mock or kvgather_sim for Decode.
  BATCH_SIZE      Number of prompts in the request.

Optional environment variables:
  LOG_DIR, PROFILE_DIR, PREFILL_HTTP_PORT, DECODE_HTTP_PORT,
  PREFILL_KV_PORT, DECODE_KV_PORT, PROMPT_TOKEN_ID, WARMUP_INPUT_LENGTH,
  WARMUP_OUTPUT_LENGTH, PROFILE_DELAY_ITERATIONS, PROFILE_DECODE_STEPS,
  PYSPY_PROFILE, PYSPY_RATE, PYSPY_BIN, PYSPY_SUDO,
  BLOCK_SIZE, MTP_SPECULATIVE_TOKENS, PREFILL_CHUNK_SIZE,
  GPU_MEMORY_UTILIZATION, STARTUP_TIMEOUT, REQUEST_TIMEOUT,
  LOCAL_SHM_DIR, LOCAL_SHM_NAMESPACE, LOCAL_SHM_TIMEOUT

Prefill runs eager with prefix caching and the mock backend. Decode always
enables npugraph_ex FULL_DECODE_ONLY Graph, --async-scheduling, and no prefix
caching. Hidden-state prefetch is disabled. A short request warms Graph before
the measured Decode-only Level1 + PipeUtilization profile starts.

Set PYSPY_PROFILE=1 to record only Decode EngineCore during measured Decode
into decode-enginecore-cpu.speedscope.json. Set PYSPY_SUDO=1 when Linux ptrace
policy requires non-interactive sudo for process attachment.

kvio is not accepted because its compact GET metadata is dynamic and cannot
run in FULL_DECODE_ONLY Graph mode.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (($# != 7)); then
    usage >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE_CLIENT="$SCRIPT_DIR/dsa_offload_pd_graph_async_profile_client.py"
MODEL="$1"
PREFILL_DEVICE="$2"
DECODE_DEVICE="$3"
INPUT_LENGTH="$4"
OUTPUT_LENGTH="$5"
IO_BACKEND="$6"
BATCH_SIZE="$7"

if [[ "$PREFILL_DEVICE" == "$DECODE_DEVICE" ]]; then
    echo "Prefill and Decode must use different physical NPUs" >&2
    exit 2
fi
if [[ "$IO_BACKEND" != "mock" && "$IO_BACKEND" != "kvgather_sim" ]]; then
    echo "IO_BACKEND must be mock or kvgather_sim" >&2
    exit 2
fi

PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-18100}"
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-18200}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-30000}"
DECODE_KV_PORT="${DECODE_KV_PORT:-30100}"
PROMPT_TOKEN_ID="${PROMPT_TOKEN_ID:-100}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
MTP_SPECULATIVE_TOKENS="${MTP_SPECULATIVE_TOKENS:-1}"
PROFILE_DELAY_ITERATIONS="${PROFILE_DELAY_ITERATIONS:-2}"
PROFILE_DECODE_STEPS="${PROFILE_DECODE_STEPS:-64}"
PYSPY_PROFILE="${PYSPY_PROFILE:-0}"
PYSPY_RATE="${PYSPY_RATE:-25}"
PYSPY_BIN="${PYSPY_BIN:-py-spy}"
PYSPY_SUDO="${PYSPY_SUDO:-0}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-7200}"
LOCAL_SHM_DIR="${LOCAL_SHM_DIR:-/dev/shm/vllm-ascend-local-kv}"
LOCAL_SHM_NAMESPACE="${LOCAL_SHM_NAMESPACE:-dsa-offload-graph-async-$$}"
LOCAL_SHM_TIMEOUT="${LOCAL_SHM_TIMEOUT:-120}"
SERVED_MODEL_NAME="dsa-offload-profile"
VERIFY_WIDTH=$((MTP_SPECULATIVE_TOKENS + 1))
MIN_PROFILE_OUTPUT_LENGTH=$(((PROFILE_DELAY_ITERATIONS + PROFILE_DECODE_STEPS + 1) * VERIFY_WIDTH))
if ((OUTPUT_LENGTH < MIN_PROFILE_OUTPUT_LENGTH)); then
    echo "OUTPUT_LENGTH must be at least $MIN_PROFILE_OUTPUT_LENGTH for" \
        "$PROFILE_DELAY_ITERATIONS delayed + $PROFILE_DECODE_STEPS profiled Decode steps" >&2
    exit 2
fi
MAX_MODEL_LEN=$((((INPUT_LENGTH + OUTPUT_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE))
WARMUP_INPUT_LENGTH="${WARMUP_INPUT_LENGTH:-$((INPUT_LENGTH < 2051 ? INPUT_LENGTH : 2051))}"
WARMUP_OUTPUT_LENGTH="${WARMUP_OUTPUT_LENGTH:-$((VERIFY_WIDTH * 2))}"

LOG_DIR="${LOG_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/dsa-pd-graph-profile.XXXXXX")}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(realpath "$LOG_DIR")"
PROFILE_DIR="${PROFILE_DIR:-$LOG_DIR/profile}"
mkdir -p "$PROFILE_DIR"
PROFILE_DIR="$(realpath "$PROFILE_DIR")"
if find "$PROFILE_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Profile directory must be empty: $PROFILE_DIR" >&2
    exit 2
fi
PYSPY_OUTPUT="$PROFILE_DIR/decode-enginecore-cpu.speedscope.json"
if [[ "$PYSPY_PROFILE" == "1" ]]; then
    command -v "$PYSPY_BIN" >/dev/null
fi

CUSTOM_OP_VENDOR_PATH="$REPO_ROOT/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
CUSTOM_OP_API_DIR="$CUSTOM_OP_VENDOR_PATH/op_api/lib"
CUSTOM_OP_API_LIB="$CUSTOM_OP_API_DIR/libcust_opapi.so"
CUSTOM_OP_KERNEL_ROOT="$CUSTOM_OP_VENDOR_PATH/op_impl/ai_core/tbe/kernel"
CUSTOM_OP_INDEX_TOOL="$REPO_ROOT/scripts/a5_kvgather_kernel_index.py"
[[ -r "$CUSTOM_OP_API_LIB" ]] || {
    echo "Custom op API library is missing: $CUSTOM_OP_API_LIB" >&2
    exit 2
}
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OP_VENDOR_PATH${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$CUSTOM_OP_API_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 "$CUSTOM_OP_INDEX_TOOL" --kernel-root "$CUSTOM_OP_KERNEL_ROOT"

PREFILL_LOG="$LOG_DIR/prefill.log"
DECODE_LOG="$LOG_DIR/decode.log"
PROFILE_RESPONSE="$LOG_DIR/profile-response.json"
PROFILE_STARTED="0"
declare -a CHILD_PIDS=()

stop_profile() {
    if [[ "$PROFILE_STARTED" == "1" ]]; then
        curl --fail --silent --show-error --request POST \
            "http://127.0.0.1:$DECODE_HTTP_PORT/stop_profile" >/dev/null
        PROFILE_STARTED="0"
    fi
}

cleanup() {
    local pid
    stop_profile || true
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
            return
        fi
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            tail -n 200 "$log" >&2
            return 1
        fi
        sleep 2
    done
    tail -n 200 "$log" >&2
    echo "Timed out waiting for $name" >&2
    return 1
}

PROFILER_CONFIG="$(python3 - "$PROFILE_DIR" "$PROFILE_DELAY_ITERATIONS" \
    "$PROFILE_DECODE_STEPS" <<'PY'
import json
import sys

print(json.dumps({
    "profiler": "torch",
    "torch_profiler_dir": sys.argv[1],
    "torch_profiler_with_stack": False,
    "ignore_frontend": True,
    "delay_iterations": int(sys.argv[2]),
    "max_iterations": int(sys.argv[3]),
}))
PY
)"
DECODE_GRAPH_CONFIG="$(python3 - "$BATCH_SIZE" "$MTP_SPECULATIVE_TOKENS" <<'PY'
import json
import sys

batch = int(sys.argv[1])
verify_width = int(sys.argv[2]) + 1
capture_sizes = set()
size = 1
while size < batch:
    capture_sizes.update((size, size * verify_width))
    size *= 2
capture_sizes.update((batch, batch * verify_width))
capture_sizes = sorted(capture_sizes)
print(json.dumps({
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": capture_sizes,
    "max_cudagraph_capture_size": capture_sizes[-1],
}))
PY
)"
DECODE_MAX_BATCHED_TOKENS="$(python3 - "$DECODE_GRAPH_CONFIG" <<'PY'
import json
import sys

print(json.loads(sys.argv[1])["max_cudagraph_capture_size"])
PY
)"
PREFILL_KV_CONFIG="$(python3 - "$PREFILL_KV_PORT" "$LOCAL_SHM_DIR" \
    "$LOCAL_SHM_NAMESPACE" "$LOCAL_SHM_TIMEOUT" <<'PY'
import json
import sys

port, shm_dir, namespace, timeout = sys.argv[1:]
print(json.dumps({
    "kv_connector": "LocalShmConnector",
    "kv_role": "kv_producer",
    "kv_port": int(port),
    "engine_id": "dsa-offload-prefill",
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": {
        "prefill": {"dp_size": 1, "tp_size": 1},
        "decode": {"dp_size": 1, "tp_size": 1},
        "shm_dir": shm_dir,
        "shm_namespace": namespace,
        "shm_timeout": float(timeout),
    },
}))
PY
)"
DECODE_KV_CONFIG="$(python3 - "$DECODE_KV_PORT" "$LOCAL_SHM_DIR" \
    "$LOCAL_SHM_NAMESPACE" "$LOCAL_SHM_TIMEOUT" <<'PY'
import json
import sys

port, shm_dir, namespace, timeout = sys.argv[1:]
print(json.dumps({
    "kv_connector": "LocalShmConnector",
    "kv_role": "kv_consumer",
    "kv_port": int(port),
    "engine_id": "dsa-offload-decode",
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": {
        "prefill": {"dp_size": 1, "tp_size": 1},
        "decode": {"dp_size": 1, "tp_size": 1},
        "shm_dir": shm_dir,
        "shm_namespace": namespace,
        "shm_timeout": float(timeout),
    },
}))
PY
)"

PREFILL_SPECULATIVE_ARGS=()
DECODE_SPECULATIVE_ARGS=()
if ((MTP_SPECULATIVE_TOKENS > 0)); then
    PREFILL_SPECULATIVE_ARGS=(
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
    )
    DECODE_SPECULATIVE_ARGS=(
        --speculative-config
        "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_SPECULATIVE_TOKENS}"
    )
fi

COMMON_ENV=(
    "PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
    "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    "OMP_PROC_BIND=false"
    "OMP_NUM_THREADS=1"
    "VLLM_ASCEND_ENABLE_DMP=0"
    "VLLM_ASCEND_ENABLE_MLAPO=0"
)
PREFILL_DSA_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":false},"dsa_offload":{"io_backend":"mock"}}'
DECODE_DSA_CONFIG="{\"ascend_compilation_config\":\
{\"enable_npugraph_ex\":true},\"dsa_offload\":\
{\"io_backend\":\"$IO_BACKEND\",\"enable_prefetch_with_hidden_states\":false}}"

echo "Starting eager Prefill on physical NPU $PREFILL_DEVICE..."
env "${COMMON_ENV[@]}" "ASCEND_RT_VISIBLE_DEVICES=$PREFILL_DEVICE" \
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
    --enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --additional-config "$PREFILL_DSA_CONFIG" \
    --kv-transfer-config "$PREFILL_KV_CONFIG" \
    "${PREFILL_SPECULATIVE_ARGS[@]}" \
    >"$PREFILL_LOG" 2>&1 &
PREFILL_PID=$!
CHILD_PIDS+=("$PREFILL_PID")

echo "Starting npugraph_ex + async Decode on physical NPU $DECODE_DEVICE..."
env "${COMMON_ENV[@]}" "ASCEND_RT_VISIBLE_DEVICES=$DECODE_DEVICE" \
    vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$DECODE_HTTP_PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$BATCH_SIZE" \
    --max-num-batched-tokens "$DECODE_MAX_BATCHED_TOKENS" \
    --block-size "$BLOCK_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code --no-enable-prefix-caching --async-scheduling \
    --compilation-config "$DECODE_GRAPH_CONFIG" \
    --additional-config "$DECODE_DSA_CONFIG" \
    --kv-transfer-config "$DECODE_KV_CONFIG" \
    --profiler-config "$PROFILER_CONFIG" \
    "${DECODE_SPECULATIVE_ARGS[@]}" \
    >"$DECODE_LOG" 2>&1 &
DECODE_PID=$!
CHILD_PIDS+=("$DECODE_PID")

wait_for_health Prefill \
    "http://127.0.0.1:$PREFILL_HTTP_PORT/health" \
    "$PREFILL_PID" "$PREFILL_LOG"
wait_for_health Decode \
    "http://127.0.0.1:$DECODE_HTTP_PORT/health" \
    "$DECODE_PID" "$DECODE_LOG"

PYSPY_ARGS=()
if [[ "$PYSPY_PROFILE" == "1" ]]; then
    PYSPY_ARGS=(
        --decode-pid "$DECODE_PID"
        --pyspy-output "$PYSPY_OUTPUT"
        --pyspy-bin "$PYSPY_BIN"
        --pyspy-rate "$PYSPY_RATE"
    )
    if [[ "$PYSPY_SUDO" == "1" ]]; then
        PYSPY_ARGS+=(--pyspy-sudo)
    fi
fi

PROFILE_STARTED="1"
python3 "$PROFILE_CLIENT" \
    --model "$SERVED_MODEL_NAME" \
    --prefill-port "$PREFILL_HTTP_PORT" \
    --decode-port "$DECODE_HTTP_PORT" \
    --batch-size "$BATCH_SIZE" \
    --warmup-input-length "$WARMUP_INPUT_LENGTH" \
    --warmup-output-length "$WARMUP_OUTPUT_LENGTH" \
    --input-length "$INPUT_LENGTH" \
    --output-length "$OUTPUT_LENGTH" \
    --token-id "$PROMPT_TOKEN_ID" \
    --timeout "$REQUEST_TIMEOUT" \
    --response-path "$PROFILE_RESPONSE" \
    "${PYSPY_ARGS[@]}"
PROFILE_STARTED="0"

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
    raise RuntimeError(f"No Ascend profiler trace found under {root}")
for trace in traces:
    analyse(str(trace))
PY

python3 - "$PROFILE_RESPONSE" "$DECODE_LOG" "$PROFILE_DIR" \
    "$BATCH_SIZE" "$INPUT_LENGTH" "$OUTPUT_LENGTH" \
    "$MODEL" "$PREFILL_DEVICE" "$DECODE_DEVICE" "$IO_BACKEND" \
    "$MTP_SPECULATIVE_TOKENS" "$PROFILE_DELAY_ITERATIONS" \
    "$PROFILE_DECODE_STEPS" "$PYSPY_PROFILE" "$PYSPY_OUTPUT" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

(
    response_path,
    decode_log_path,
    profile_dir,
    batch_size,
    input_length,
    output_length,
    model,
    prefill_device,
    decode_device,
    io_backend,
    mtp_speculative_tokens,
    profile_delay_iterations,
    profile_decode_steps,
    pyspy_profile,
    pyspy_output,
) = sys.argv[1:]
batch_size = int(batch_size)
input_length = int(input_length)
output_length = int(output_length)
mtp_speculative_tokens = int(mtp_speculative_tokens)
profile_delay_iterations = int(profile_delay_iterations)
profile_decode_steps = int(profile_decode_steps)
pyspy_profile = pyspy_profile == "1"

responses = json.loads(Path(response_path).read_text(encoding="utf-8"))
if len(responses) != batch_size:
    raise RuntimeError(f"Expected {batch_size} responses, got {len(responses)}")
for response in responses:
    choices = response["choices"]
    usage = response["usage"]
    if len(choices) != 1:
        raise RuntimeError(f"Expected one choice, got {len(choices)}")
    if usage["prompt_tokens"] != input_length:
        raise RuntimeError(f"Unexpected prompt token count: {usage}")
    if usage["completion_tokens"] != output_length:
        raise RuntimeError(f"Unexpected completion token count: {usage}")

decode_log = Path(decode_log_path).read_text(encoding="utf-8", errors="replace")
graph_marker = (
    "DSA_OFFLOAD_KVGATHER_SIM_GRAPH_ACTIVE"
    if io_backend == "kvgather_sim"
    else "DSA_OFFLOAD_MOCK_GRAPH_ACTIVE"
)
required_log_markers = (
    graph_marker,
    "Replaying aclgraph",
    "Max profiling iterations reached",
)
missing_log_markers = [marker for marker in required_log_markers if marker not in decode_log]
if missing_log_markers:
    raise RuntimeError("Decode log is missing markers: " + ", ".join(missing_log_markers))

if pyspy_profile:
    json.loads(Path(pyspy_output).read_text(encoding="utf-8"))

outputs = sorted(Path(profile_dir).rglob("ASCEND_PROFILER_OUTPUT"))
if not outputs:
    raise RuntimeError("ASCEND_PROFILER_OUTPUT was not generated")
operator_text = ""
for output in outputs:
    kernel_details = output / "kernel_details.csv"
    with kernel_details.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream))
    if len(columns) < 40:
        raise RuntimeError(
            f"{kernel_details} has {len(columns)} columns; expected Level1 profiling"
        )
    for name in ("op_statistic.csv", "api_statistic.csv"):
        path = output / name
        if not path.is_file():
            raise RuntimeError(f"Missing profiler artifact: {path}")
    json.loads((output / "trace_view.json").read_text(encoding="utf-8"))
    for name in ("kernel_details.csv", "operator_details.csv", "op_statistic.csv"):
        path = output / name
        if path.is_file():
            operator_text += path.read_text(encoding="utf-8-sig", errors="replace")

normalized = re.sub(r"[^a-z0-9]", "", operator_text.lower())
required = ["lookupupdate", "sparseflashattention"]
if io_backend == "kvgather_sim":
    required.append("asukvgather")
missing = [name for name in required if name not in normalized]
if missing:
    raise RuntimeError("Profile is missing operators: " + ", ".join(missing))

print(json.dumps({
    "model": model,
    "prefill_device": prefill_device,
    "decode_device": decode_device,
    "io_backend": io_backend,
    "batch_size": batch_size,
    "input_length": input_length,
    "output_length": output_length,
    "graph_mode": "FULL_DECODE_ONLY",
    "npugraph_ex": True,
    "hidden_state_prefetch": False,
    "mtp_speculative_tokens": mtp_speculative_tokens,
    "profile_delay_iterations": profile_delay_iterations,
    "profile_decode_steps": profile_decode_steps,
    "pyspy_profile": pyspy_profile,
    "pyspy_output": pyspy_output if pyspy_profile else None,
    "async_scheduling": True,
    "profile_scope": "decode_only",
    "profile_outputs": [str(path) for path in outputs],
}, indent=2, sort_keys=True))
PY

echo "logs=$LOG_DIR"
echo "profile=$PROFILE_DIR"
