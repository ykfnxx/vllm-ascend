#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#
# Exercise the framework-side DSA Offload path with a small glm_moe_dsa model.
#
# The default P/D + mock scenario validates process isolation, connector handoff,
# DSA lookup/update, Hot Cache use, and Sparse Flash Attention execution.
# Select --io-backend kvio to additionally exercise real block PUT/token GET.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_MODEL="$REPO_ROOT/../weights/tiny-random-glm-moe-dsa"
if [[ ! -d "$DEFAULT_MODEL" ]]; then
    DEFAULT_MODEL="tiny-random/glm-moe-dsa"
fi

MODEL="$DEFAULT_MODEL"
SERVED_MODEL_NAME="glm-moe-dsa"
SCENARIO="pd"
CONNECTOR="mooncake"
IO_BACKEND="mock"
KVIO_MODEL_ID="0"
HOST_IP=""
IFNAME=""
PREFILL_DEVICE="0"
DECODE_DEVICE="1"
BOTH_DEVICE="0"
PREFILL_HTTP_PORT="18100"
DECODE_HTTP_PORT="18200"
BOTH_HTTP_PORT="18300"
PROXY_HTTP_PORT="18000"
PREFILL_KV_PORT="30000"
DECODE_KV_PORT="30100"
BOTH_KV_PORT="30200"
PROMPT_TOKENS="2333"
PROMPT_TOKEN_ID="100"
MAX_TOKENS="4"
MAX_MODEL_LEN="4096"
BLOCK_SIZE="128"
MTP_SPECULATIVE_TOKENS="0"
ENABLE_PREFETCH_WITH_HIDDEN_STATES="0"
PREFETCH_TOP_K="2048"
GPU_MEMORY_UTILIZATION="0.50"
STARTUP_TIMEOUT="900"
LOG_DIR=""
VERIFY_PATH="0"
RUN_UNIT_TESTS="0"
SKIP_CONCURRENT="0"
LOCAL_SHM_DIR="/dev/shm/vllm-ascend-local-kv"
LOCAL_SHM_NAMESPACE="dsa-offload-probe"
LOCAL_SHM_TIMEOUT="120"

usage() {
    cat <<'EOF'
Usage:
  dsa_offload_probe.sh --host-ip IP --ifname NIC [options]

Scenarios:
  --scenario pd               Same-node Prefill/Decode engines plus proxy.
                              This is the default and uses two NPUs.
  --scenario both             One kv_both engine. Also submits two concurrent
                              requests as a mixed-batch scheduling candidate.

Core options:
  --model MODEL               Local model path or Hugging Face ID.
  --served-model-name NAME    OpenAI API name. Default: glm-moe-dsa
  --connector TYPE            mooncake, local-shm, or none. Default: mooncake
                              local-shm is for split P/D on one host/container;
                              none is for the single-engine both scenario.
  --io-backend BACKEND        mock, kvio, or kvgather_sim. Default: mock
  --kvio-model-id ID          Non-negative KVIO model namespace. Default: 0
  --host-ip IP                Local IP used by Mooncake/HCCL.
  --ifname NIC                Network interface owning --host-ip.
  --prefill-device ID         P/D Prefill physical NPU. Default: 0
  --decode-device ID          P/D Decode physical NPU. Default: 1
  --both-device ID            kv_both physical NPU. Default: 0
  --prompt-tokens N           Non-block-aligned history workload. Default: 2333
  --prompt-token-id ID        Repeated vocabulary token ID. Default: 100
  --max-tokens N              Decode tokens for the main workload. Default: 4
  --mtp-speculative-tokens N  Enable MTP with N draft tokens, range [0, 15].
  --enable-prefetch-with-hidden-states
                              Enable grouped hidden-state prefetch on Decode.
  --prefetch-top-k N          Predicted Top-K width, range [128, 2048].
                              Default: 2048
  --max-model-len N           Context limit. Default: 4096
  --block-size N               Force one known block size. Default: 128
  --gpu-memory-utilization F  Per-engine NPU memory fraction. Default: 0.50
  --startup-timeout SEC       Per-service startup timeout. Default: 900
  --log-dir DIR               Keep artifacts in DIR. Default: /tmp directory
  --verify-path               Profile runtime and require DSA lookup + SFA;
                              with kvio also require PUT/GET operator evidence.
  --run-unit-tests             Run tests/ut/dsa_offload before the NPU probe.
  --skip-concurrent            In kv_both mode, skip the concurrent workload.

LocalShm options:
  --local-shm-dir DIR         Shared-memory root. Default:
                              /dev/shm/vllm-ascend-local-kv
  --local-shm-namespace NAME  Isolation namespace. Default: dsa-offload-probe
  --local-shm-timeout SEC     Manifest wait timeout. Default: 120

Port options:
  --prefill-http-port PORT    Default: 18100
  --decode-http-port PORT     Default: 18200
  --both-http-port PORT       Default: 18300
  --proxy-http-port PORT      Default: 18000
  --prefill-kv-port PORT      Default: 30000
  --decode-kv-port PORT       Default: 30100
  --both-kv-port PORT         Default: 30200
  -h, --help                  Show this help.

Workloads:
  1. block-aligned 2048-token prompt: full-block publication and lookup;
  2. non-aligned long prompt: partial-tail handoff plus multi-step Decode;
  3. repeated long prompt: stable block-hash/storage-key lifecycle;
  4. kv_both only: overlapping requests to make mixed Prefill/Decode possible.

Validation boundary:
  Prefix caching stays disabled so repeated prompts exercise DSA Offload rather
  than vLLM prefix reuse; DSA block hashes are generated independently.
  mock is intentionally the default. With a split P/D connector it can still
  exercise the partial-tail handoff, but it does not perform capacity-layer
  full-block PUT or token GET. With --connector none it validates local cache
  promotion only at the control-path level. kvio adds real capacity-layer
  PUT/GET. kvgather_sim invokes ASU KV Gather with synthetic zero source
  blocks. Output accuracy still needs a known-good baseline comparison.
  With hidden-state prefetch enabled, --verify-path additionally requires the
  predicted LightningIndexer and the MTP prefetch lookup when MTP is enabled.
EOF
}

require_value() {
    if (($# < 2)); then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
    fi
}

require_uint() {
    local value="$1"
    local name="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$name must be a non-negative integer, got: $value" >&2
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
        --scenario)
            require_value "$@"
            SCENARIO="$2"
            shift 2
            ;;
        --io-backend)
            require_value "$@"
            IO_BACKEND="$2"
            shift 2
            ;;
        --connector)
            require_value "$@"
            CONNECTOR="$2"
            shift 2
            ;;
        --kvio-model-id)
            require_value "$@"
            KVIO_MODEL_ID="$2"
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
        --both-device)
            require_value "$@"
            BOTH_DEVICE="$2"
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
        --both-http-port)
            require_value "$@"
            BOTH_HTTP_PORT="$2"
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
        --both-kv-port)
            require_value "$@"
            BOTH_KV_PORT="$2"
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
        --mtp-speculative-tokens)
            require_value "$@"
            MTP_SPECULATIVE_TOKENS="$2"
            shift 2
            ;;
        --enable-prefetch-with-hidden-states)
            ENABLE_PREFETCH_WITH_HIDDEN_STATES="1"
            shift
            ;;
        --prefetch-top-k)
            require_value "$@"
            PREFETCH_TOP_K="$2"
            shift 2
            ;;
        --max-model-len)
            require_value "$@"
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --block-size)
            require_value "$@"
            BLOCK_SIZE="$2"
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
        --verify-path)
            VERIFY_PATH="1"
            shift
            ;;
        --run-unit-tests)
            RUN_UNIT_TESTS="1"
            shift
            ;;
        --skip-concurrent)
            SKIP_CONCURRENT="1"
            shift
            ;;
        --local-shm-dir)
            require_value "$@"
            LOCAL_SHM_DIR="$2"
            shift 2
            ;;
        --local-shm-namespace)
            require_value "$@"
            LOCAL_SHM_NAMESPACE="$2"
            shift 2
            ;;
        --local-shm-timeout)
            require_value "$@"
            LOCAL_SHM_TIMEOUT="$2"
            shift 2
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

if [[ "$SCENARIO" != "pd" && "$SCENARIO" != "both" ]]; then
    echo "--scenario must be 'pd' or 'both'." >&2
    exit 2
fi
if [[ "$IO_BACKEND" != "mock" && "$IO_BACKEND" != "kvio" && \
      "$IO_BACKEND" != "kvgather_sim" ]]; then
    echo "--io-backend must be 'mock', 'kvio', or 'kvgather_sim'." >&2
    exit 2
fi
if [[ "$CONNECTOR" != "mooncake" && "$CONNECTOR" != "local-shm" && "$CONNECTOR" != "none" ]]; then
    echo "--connector must be 'mooncake', 'local-shm', or 'none'." >&2
    exit 2
fi
if [[ "$SCENARIO" == "pd" && "$CONNECTOR" == "none" ]]; then
    echo "Split P/D requires --connector mooncake or local-shm." >&2
    exit 2
fi
if [[ "$SCENARIO" == "both" && "$CONNECTOR" == "local-shm" ]]; then
    echo "local-shm supports split P/D only; use --connector none for local both." >&2
    exit 2
fi
if [[ "$CONNECTOR" == "local-shm" && "$LOCAL_SHM_DIR" != /* ]]; then
    echo "--local-shm-dir must be an absolute path." >&2
    exit 2
fi
if [[ -z "$HOST_IP" || -z "$IFNAME" ]]; then
    echo "--host-ip and --ifname are required." >&2
    usage >&2
    exit 2
fi
if [[ "$SCENARIO" == "pd" && "$PREFILL_DEVICE" == "$DECODE_DEVICE" ]]; then
    echo "Prefill and Decode must use different physical NPU IDs." >&2
    exit 2
fi

for value_and_name in \
    "$KVIO_MODEL_ID:kvio-model-id" \
    "$PROMPT_TOKENS:prompt-tokens" \
    "$PROMPT_TOKEN_ID:prompt-token-id" \
    "$MAX_TOKENS:max-tokens" \
    "$MTP_SPECULATIVE_TOKENS:mtp-speculative-tokens" \
    "$PREFETCH_TOP_K:prefetch-top-k" \
    "$MAX_MODEL_LEN:max-model-len" \
    "$BLOCK_SIZE:block-size" \
    "$STARTUP_TIMEOUT:startup-timeout"; do
    require_uint "${value_and_name%%:*}" "${value_and_name#*:}"
done
if ! [[ "$LOCAL_SHM_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || [[ "$LOCAL_SHM_TIMEOUT" =~ ^0+([.]0+)?$ ]]; then
    echo "local-shm-timeout must be positive." >&2
    exit 2
fi

if ((MTP_SPECULATIVE_TOKENS > 15)); then
    echo "mtp-speculative-tokens must be in [0, 15]." >&2
    exit 2
fi
if ((PREFETCH_TOP_K < 128 || PREFETCH_TOP_K > 2048)); then
    echo "prefetch-top-k must be in [128, 2048]." >&2
    exit 2
fi
if ((BLOCK_SIZE == 0 || MAX_TOKENS == 0)); then
    echo "block-size and max-tokens must be positive." >&2
    exit 2
fi
if [[ "$BLOCK_SIZE" != "32" && "$BLOCK_SIZE" != "64" && "$BLOCK_SIZE" != "128" ]]; then
    echo "block-size must be one of 32, 64, or 128 on Ascend." >&2
    exit 2
fi

QUERY_WIDTH=2048
ALIGNED_PROMPT_TOKENS=$((((QUERY_WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE))
TAIL_PROMPT_TOKENS="$PROMPT_TOKENS"
if ((TAIL_PROMPT_TOKENS < QUERY_WIDTH + 3)); then
    TAIL_PROMPT_TOKENS=$((QUERY_WIDTH + 3))
fi
if ((TAIL_PROMPT_TOKENS % BLOCK_SIZE == 0)); then
    TAIL_PROMPT_TOKENS=$((TAIL_PROMPT_TOKENS + 3))
fi
if ((ALIGNED_PROMPT_TOKENS + 1 > MAX_MODEL_LEN)); then
    echo "max-model-len is too small for the aligned $ALIGNED_PROMPT_TOKENS-token workload." >&2
    exit 2
fi
if ((TAIL_PROMPT_TOKENS + MAX_TOKENS > MAX_MODEL_LEN)); then
    echo "The non-aligned workload needs max-model-len >= $((TAIL_PROMPT_TOKENS + MAX_TOKENS))." >&2
    exit 2
fi

for command_name in vllm python3 curl git; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command not found: $command_name" >&2
        exit 2
    fi
done

if [[ "$RUN_UNIT_TESTS" == "1" ]]; then
    echo "Running framework-side DSA Offload unit tests..."
    (
        cd "$REPO_ROOT"
        python3 -m pytest -q \
            tests/ut/dsa_offload \
            tests/ut/kv_offload/test_local_shm_connector.py
    )
fi

echo "Running DSA Offload preflight for model: $MODEL"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$REPO_ROOT" \
    "$MODEL" \
    "$IO_BACKEND" \
    "$PROMPT_TOKEN_ID" \
    "$MTP_SPECULATIVE_TOKENS" \
    "$ENABLE_PREFETCH_WITH_HIDDEN_STATES" <<'PY'
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
model_name = sys.argv[2]
io_backend = sys.argv[3]
prompt_token_id = int(sys.argv[4])
mtp_tokens = int(sys.argv[5])
prefetch_enabled = bool(int(sys.argv[6]))

model_path = Path(model_name)
if model_path.is_dir() and (model_path / "config.json").is_file():
    with (model_path / "config.json").open(encoding="utf-8") as config_file:
        raw_config = json.load(config_file)
else:
    from transformers import AutoConfig

    raw_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    ).to_dict()

text_config = raw_config.get("text_config") or raw_config
model_type = text_config.get("model_type")
index_topk = text_config.get("index_topk")
if model_type != "glm_moe_dsa":
    raise SystemExit(
        f"DSA Offload requires model_type='glm_moe_dsa', got {model_type!r}"
    )
if index_topk != 2048:
    raise SystemExit(f"DSA Offload requires index_topk=2048, got {index_topk!r}")

vocab_size = text_config.get("vocab_size")
if isinstance(vocab_size, int) and not 0 <= prompt_token_id < vocab_size:
    raise SystemExit(
        f"prompt-token-id={prompt_token_id} is outside vocab_size={vocab_size}"
    )
if mtp_tokens and not text_config.get("num_nextn_predict_layers", 0):
    raise SystemExit("MTP was requested but the model has no next-token prediction layer")

hidden_size = text_config.get("hidden_size")
if isinstance(hidden_size, int) and hidden_size <= 8:
    print(
        "WARNING: this tiny fixture has hidden_size<=8. Some A5 MlaPrologV3 "
        "builds reject He=8; use --model with a hardware-compatible small "
        "checkpoint if startup fails.",
        file=sys.stderr,
    )

import torch
import vllm_ascend

package_path = Path(vllm_ascend.__file__).resolve()
if repo_root not in package_path.parents:
    raise SystemExit(
        f"vllm_ascend resolves to {package_path}, not the checkout {repo_root}"
    )

importlib.import_module("vllm_ascend.vllm_ascend_C")
namespace = torch.ops._C_ascend
required_ops = [
    "dsa_offload_lookup_update",
    "dsa_offload_lookup_update_batch",
    "dsa_sparse_turbo_lookup_update_batch",
]
if prefetch_enabled:
    required_ops.extend(
        [
            "dsa_sparse_turbo_prefetch_lookup_update_batch",
            "npu_lightning_indexer_hi_cached",
            "npu_scatter_nd_update_mean",
            "prefetch_qli_fusion",
        ]
    )
if io_backend == "kvio":
    required_ops.extend(["npu_get_put_batch", "npu_send_wait"])
    rdma_kv_ops = importlib.import_module("rdma_kv_ops")
    if not hasattr(rdma_kv_ops, "aiv_init"):
        raise SystemExit("rdma_kv_ops does not expose aiv_init")
elif io_backend == "kvgather_sim":
    required_ops.append("asu_kv_gather")

missing = [name for name in required_ops if not hasattr(namespace, name)]
if missing:
    raise SystemExit(
        "Loaded _C_ascend is missing required operators: " + ", ".join(missing)
    )

print(
    json.dumps(
        {
            "model_type": model_type,
            "index_topk": index_topk,
            "hidden_size": hidden_size,
            "num_hidden_layers": text_config.get("num_hidden_layers"),
            "num_nextn_predict_layers": text_config.get(
                "num_nextn_predict_layers", 0
            ),
            "io_backend": io_backend,
            "enable_prefetch_with_hidden_states": prefetch_enabled,
            "vllm_ascend": str(package_path),
            "native_ops": required_ops,
        },
        sort_keys=True,
    )
)
PY

if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dsa-offload-probe.XXXXXX")"
else
    mkdir -p "$LOG_DIR"
    LOG_DIR="$(cd "$LOG_DIR" && pwd)"
fi

HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
BRANCH_NAME="$(git -C "$REPO_ROOT" branch --show-current)"
python3 - \
    "$LOG_DIR/manifest.json" \
    "$BRANCH_NAME" \
    "$HEAD_SHA" \
    "$MODEL" \
    "$SCENARIO" \
    "$CONNECTOR" \
    "$IO_BACKEND" \
    "$KVIO_MODEL_ID" \
    "$BLOCK_SIZE" \
    "$ALIGNED_PROMPT_TOKENS" \
    "$TAIL_PROMPT_TOKENS" \
    "$MAX_TOKENS" \
    "$MTP_SPECULATIVE_TOKENS" \
    "$ENABLE_PREFETCH_WITH_HIDDEN_STATES" \
    "$PREFETCH_TOP_K" <<'PY'
import json
import sys

(
    output_path,
    branch,
    head,
    model,
    scenario,
    connector,
    io_backend,
    kvio_model_id,
    block_size,
    aligned_prompt_tokens,
    tail_prompt_tokens,
    max_tokens,
    mtp_speculative_tokens,
    enable_prefetch_with_hidden_states,
    prefetch_top_k,
) = sys.argv[1:]
manifest = {
    "branch": branch,
    "head": head,
    "model": model,
    "scenario": scenario,
    "connector": connector,
    "io_backend": io_backend,
    "kvio_model_id": int(kvio_model_id),
    "block_size": int(block_size),
    "aligned_prompt_tokens": int(aligned_prompt_tokens),
    "tail_prompt_tokens": int(tail_prompt_tokens),
    "max_tokens": int(max_tokens),
    "mtp_speculative_tokens": int(mtp_speculative_tokens),
    "enable_prefetch_with_hidden_states": bool(
        int(enable_prefetch_with_hidden_states)
    ),
    "prefetch_top_k": int(prefetch_top_k),
}
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(manifest, output_file, indent=2, sort_keys=True)
PY

declare -a CHILD_PIDS=()
declare -a PROFILE_PORTS=()
PROFILE_STARTED="0"

stop_profiles() {
    local port
    if [[ "$PROFILE_STARTED" != "1" ]]; then
        return 0
    fi
    for port in "${PROFILE_PORTS[@]:-}"; do
        if ! curl --fail --silent --show-error \
            --request POST \
            "http://127.0.0.1:$port/stop_profile" \
            >/dev/null; then
            echo "Failed to stop profiler on HTTP port $port." >&2
        fi
    done
    PROFILE_STARTED="0"
}

cleanup() {
    local pid
    stop_profiles || true
    for pid in "${CHILD_PIDS[@]:-}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done
    for pid in "${CHILD_PIDS[@]:-}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done
    echo "Artifacts kept in: $LOG_DIR"
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
        if curl --fail --silent "$health_url" >/dev/null 2>&1; then
            echo "$service_name is ready: $health_url"
            return 0
        fi
        if ! kill -0 "$service_pid" >/dev/null 2>&1; then
            echo "$service_name exited before becoming ready." >&2
            tail -n 160 "$service_log" >&2 || true
            return 1
        fi
        sleep 2
    done

    echo "Timed out waiting for $service_name." >&2
    tail -n 160 "$service_log" >&2 || true
    return 1
}

COMMON_NETWORK_ENV=(
    "PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "VLLM_HOST_IP=$HOST_IP"
    "HCCL_IF_IP=$HOST_IP"
    "GLOO_SOCKET_IFNAME=$IFNAME"
    "TP_SOCKET_IFNAME=$IFNAME"
    "HCCL_SOCKET_IFNAME=$IFNAME"
    "MC_TCP_BIND_ADDRESS=$HOST_IP"
    "OMP_PROC_BIND=false"
    "OMP_NUM_THREADS=1"
)

LAST_PID=""
launch_server() {
    local service_name="$1"
    local device_id="$2"
    local http_port="$3"
    local kv_role="$4"
    local kv_port="$5"
    local engine_id="$6"
    local service_log="$7"
    local profile_dir="$8"
    local speculative_tokens="$9"
    local kv_config
    local profiler_config
    local prefetch_enabled="false"
    local dsa_config
    local -a kv_transfer_args=()
    local -a profiler_args=()
    local -a speculative_args=()

    if [[ "$CONNECTOR" != "none" ]]; then
        kv_config="$(python3 - \
            "$CONNECTOR" \
            "$kv_role" \
            "$kv_port" \
            "$engine_id" \
            "$LOCAL_SHM_DIR" \
            "$LOCAL_SHM_NAMESPACE" \
            "$LOCAL_SHM_TIMEOUT" <<'PY'
import json
import sys

(
    connector,
    role,
    port,
    engine_id,
    shm_dir,
    shm_namespace,
    shm_timeout,
) = sys.argv[1:]
extra = {
    "prefill": {"dp_size": 1, "tp_size": 1},
    "decode": {"dp_size": 1, "tp_size": 1},
}
if connector == "local-shm":
    extra.update(
        shm_dir=shm_dir,
        shm_namespace=shm_namespace,
        shm_timeout=float(shm_timeout),
    )
config = {
    "kv_connector": (
        "MooncakeConnectorV1"
        if connector == "mooncake"
        else "LocalShmConnector"
    ),
    "kv_role": role,
    "kv_port": int(port),
    "engine_id": engine_id,
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": extra,
}
print(json.dumps(config, separators=(",", ":")))
PY
)"
        kv_transfer_args+=(--kv-transfer-config "$kv_config")
    fi
    if ((speculative_tokens > 0)); then
        speculative_args+=(
            --speculative-config
            "{\"method\":\"mtp\",\"num_speculative_tokens\":$speculative_tokens}"
        )
    fi
    if [[ "$kv_role" != "kv_producer" \
        && "$ENABLE_PREFETCH_WITH_HIDDEN_STATES" == "1" ]]; then
        prefetch_enabled="true"
    fi
    dsa_config="{\"ascend_compilation_config\":{\"enable_npugraph_ex\":false},\"dsa_offload\":{\"io_backend\":\"$IO_BACKEND\",\"kvio_model_id\":$KVIO_MODEL_ID,\"enable_prefetch_with_hidden_states\":$prefetch_enabled,\"prefetch_top_k\":$PREFETCH_TOP_K}}"
    if [[ "$VERIFY_PATH" == "1" ]] \
        && [[ "$kv_role" != "kv_producer" || "$IO_BACKEND" == "kvio" ]]; then
        mkdir -p "$profile_dir"
        profiler_config="$(python3 - "$profile_dir" <<'PY'
import json
import sys

print(
    json.dumps(
        {
            "profiler": "torch",
            "torch_profiler_dir": sys.argv[1],
            "torch_profiler_with_stack": False,
        }
    )
)
PY
)"
        profiler_args+=(--profiler-config "$profiler_config")
        PROFILE_PORTS+=("$http_port")
    fi

    echo "Starting $service_name on physical NPU $device_id..."
    echo "  hidden_state_prefetch=$prefetch_enabled prefetch_top_k=$PREFETCH_TOP_K"
    env \
        "${COMMON_NETWORK_ENV[@]}" \
        "ASCEND_RT_VISIBLE_DEVICES=$device_id" \
        vllm serve "$MODEL" \
        --host 0.0.0.0 \
        --port "$http_port" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --quantization ascend \
        --tensor-parallel-size 1 \
        --block-size "$BLOCK_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs 2 \
        --max-num-batched-tokens "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --seed 0 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --additional-config "$dsa_config" \
        "${kv_transfer_args[@]}" \
        "${speculative_args[@]}" \
        "${profiler_args[@]}" \
        >"$service_log" 2>&1 &
    LAST_PID=$!
    CHILD_PIDS+=("$LAST_PID")
}

PREFILL_LOG="$LOG_DIR/prefill.log"
DECODE_LOG="$LOG_DIR/decode.log"
BOTH_LOG="$LOG_DIR/both.log"
PROXY_LOG="$LOG_DIR/proxy.log"
PREFILL_PROFILE_DIR="$LOG_DIR/prefill-profile"
DECODE_PROFILE_DIR="$LOG_DIR/decode-profile"
BOTH_PROFILE_DIR="$LOG_DIR/both-profile"

if [[ "$VERIFY_PATH" == "1" ]]; then
    for profile_dir in \
        "$PREFILL_PROFILE_DIR" \
        "$DECODE_PROFILE_DIR" \
        "$BOTH_PROFILE_DIR"; do
        if [[ -d "$profile_dir" ]] \
            && find "$profile_dir" -mindepth 1 -print -quit | grep --quiet .; then
            echo "Profile directory is not empty: $profile_dir" >&2
            echo "Use a new --log-dir so stale data cannot satisfy --verify-path." >&2
            exit 2
        fi
    done
fi

if [[ "$SCENARIO" == "pd" ]]; then
    PREFILL_SPECULATIVE_TOKENS="0"
    if ((MTP_SPECULATIVE_TOKENS > 0)); then
        PREFILL_SPECULATIVE_TOKENS="1"
    fi
    launch_server \
        "Prefill" \
        "$PREFILL_DEVICE" \
        "$PREFILL_HTTP_PORT" \
        "kv_producer" \
        "$PREFILL_KV_PORT" \
        "dsa-offload-prefill" \
        "$PREFILL_LOG" \
        "$PREFILL_PROFILE_DIR" \
        "$PREFILL_SPECULATIVE_TOKENS"
    PREFILL_PID="$LAST_PID"

    launch_server \
        "Decode" \
        "$DECODE_DEVICE" \
        "$DECODE_HTTP_PORT" \
        "kv_consumer" \
        "$DECODE_KV_PORT" \
        "dsa-offload-decode" \
        "$DECODE_LOG" \
        "$DECODE_PROFILE_DIR" \
        "$MTP_SPECULATIVE_TOKENS"
    DECODE_PID="$LAST_PID"

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
    env "PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
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
    REQUEST_BASE_URL="http://127.0.0.1:$PROXY_HTTP_PORT"
else
    launch_server \
        "Both" \
        "$BOTH_DEVICE" \
        "$BOTH_HTTP_PORT" \
        "kv_both" \
        "$BOTH_KV_PORT" \
        "dsa-offload-both" \
        "$BOTH_LOG" \
        "$BOTH_PROFILE_DIR" \
        "$MTP_SPECULATIVE_TOKENS"
    BOTH_PID="$LAST_PID"
    wait_for_health \
        "Both" \
        "http://127.0.0.1:$BOTH_HTTP_PORT/health" \
        "$BOTH_PID" \
        "$BOTH_LOG"
    REQUEST_BASE_URL="http://127.0.0.1:$BOTH_HTTP_PORT"
fi

if [[ "$VERIFY_PATH" == "1" ]]; then
    echo "Starting runtime profiler(s)..."
    for port in "${PROFILE_PORTS[@]}"; do
        curl --fail --silent --show-error \
            --request POST \
            "http://127.0.0.1:$port/start_profile" \
            >/dev/null
    done
    PROFILE_STARTED="1"
fi

echo "Submitting aligned, partial-tail, and repeated-history workloads..."
python3 - \
    "$REQUEST_BASE_URL" \
    "$SERVED_MODEL_NAME" \
    "$LOG_DIR" \
    "$ALIGNED_PROMPT_TOKENS" \
    "$TAIL_PROMPT_TOKENS" \
    "$PROMPT_TOKEN_ID" \
    "$MAX_TOKENS" \
    "$SCENARIO" \
    "$SKIP_CONCURRENT" <<'PY'
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

(
    base_url,
    model_name,
    output_dir_raw,
    aligned_tokens_raw,
    tail_tokens_raw,
    token_id_raw,
    max_tokens_raw,
    scenario,
    skip_concurrent_raw,
) = sys.argv[1:]
output_dir = Path(output_dir_raw)
aligned_tokens = int(aligned_tokens_raw)
tail_tokens = int(tail_tokens_raw)
token_id = int(token_id_raw)
max_tokens = int(max_tokens_raw)
skip_concurrent = bool(int(skip_concurrent_raw))


def submit(name: str, prompt_tokens: int, output_tokens: int) -> dict:
    payload = {
        "model": model_name,
        "prompt": [token_id] * prompt_tokens,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    request_path = output_dir / f"request-{name}.json"
    response_path = output_dir / f"response-{name}.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            status = response.status
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read().decode("utf-8", errors="replace")
    response_path.write_text(body, encoding="utf-8")
    if not 200 <= status < 300:
        raise RuntimeError(f"{name} failed with HTTP {status}: {body[:1000]}")
    parsed = json.loads(body)
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(f"{name} response has no usage object")
    if usage.get("prompt_tokens") != prompt_tokens:
        raise RuntimeError(
            f"{name} prompt token mismatch: {usage.get('prompt_tokens')} != "
            f"{prompt_tokens}"
        )
    if usage.get("completion_tokens") != output_tokens:
        raise RuntimeError(
            f"{name} completion token mismatch: "
            f"{usage.get('completion_tokens')} != {output_tokens}"
        )
    summary = {
        "name": name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "elapsed_seconds": round(time.monotonic() - start, 3),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


summaries = [
    submit("aligned-full-blocks", aligned_tokens, 1),
    submit("partial-tail-multistep", tail_tokens, max_tokens),
    submit("repeat-history", tail_tokens, min(max_tokens, 2)),
]

if scenario == "both" and not skip_concurrent:
    concurrent_tokens = max(max_tokens, 8)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            submit,
            "concurrent-long",
            tail_tokens,
            concurrent_tokens,
        )
        time.sleep(0.1)
        second = executor.submit(
            submit,
            "concurrent-new-prefill",
            aligned_tokens,
            2,
        )
        summaries.extend((first.result(), second.result()))

(output_dir / "workload-summary.json").write_text(
    json.dumps(summaries, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY

if [[ "$VERIFY_PATH" == "1" ]]; then
    echo "Stopping and analyzing runtime profiler(s)..."
    stop_profiles
    for profile_dir in \
        "$PREFILL_PROFILE_DIR" \
        "$DECODE_PROFILE_DIR" \
        "$BOTH_PROFILE_DIR"; do
        if [[ ! -d "$profile_dir" ]]; then
            continue
        fi
        python3 - "$profile_dir" <<'PY'
import sys
import time
from pathlib import Path

from torch_npu.profiler.profiler import analyse

profile_root = Path(sys.argv[1])
deadline = time.monotonic() + 60
trace_directories = []
while time.monotonic() < deadline:
    trace_directories = sorted(
        path for path in profile_root.rglob("*_ascend_pt") if path.is_dir()
    )
    if trace_directories:
        break
    time.sleep(1)
if not trace_directories:
    raise SystemExit(f"No Ascend profiler trace found under {profile_root}")
for trace_directory in trace_directories:
    analyse(str(trace_directory))
PY
    done

    python3 - \
        "$SCENARIO" \
        "$IO_BACKEND" \
        "$ENABLE_PREFETCH_WITH_HIDDEN_STATES" \
        "$MTP_SPECULATIVE_TOKENS" \
        "$PREFILL_PROFILE_DIR" \
        "$DECODE_PROFILE_DIR" \
        "$BOTH_PROFILE_DIR" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

(
    scenario,
    io_backend,
    prefetch_enabled_raw,
    mtp_speculative_tokens_raw,
    prefill_raw,
    decode_raw,
    both_raw,
) = sys.argv[1:]
prefetch_enabled = bool(int(prefetch_enabled_raw))
mtp_enabled = int(mtp_speculative_tokens_raw) > 0
profile_files = {
    "Prefill": Path(prefill_raw),
    "Decode": Path(decode_raw),
    "Both": Path(both_raw),
}
allowed_names = {
    "kernel_details.csv",
    "operator_details.csv",
    "op_statistic.csv",
}


def read_profile(name: str) -> str:
    root = profile_files[name]
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name in allowed_names
    ]
    if not files:
        raise RuntimeError(f"No analyzed operator CSV found under {root}")
    return re.sub(
        r"[^a-z0-9]+",
        "",
        "\n".join(
            path.read_text(encoding="utf-8", errors="replace").lower()
            for path in files
        ),
    )


def require_any(profile: str, names: tuple[str, ...], label: str) -> None:
    if not any(name in profile for name in names):
        raise RuntimeError(f"Profiler evidence missing: {label} ({names})")


decode_name = "Decode" if scenario == "pd" else "Both"
decode_profile = read_profile(decode_name)
require_any(
    decode_profile,
    (
        "dsaoffloadlookupupdatebatch",
        "dsaoffloadlookupupdate",
        "aclndsaoffloadlookupupdatebatch",
        "aclndsaoffloadlookupupdate",
        "dsasparseturbolookupupdatebatch",
        "aclndsasparseturbolookupupdatebatch",
    ),
    f"{decode_name} DSA lookup/update",
)
require_any(
    decode_profile,
    ("sparseflashattention", "nputsparseflashattention"),
    f"{decode_name} Sparse Flash Attention",
)

if prefetch_enabled and mtp_enabled:
    require_any(
        decode_profile,
        (
            "dsasparseturboprefetchlookupupdatebatch",
            "aclndsasparseturboprefetchlookupupdatebatch",
        ),
        f"{decode_name} DSA predicted prefetch lookup/update",
    )
if prefetch_enabled:
    require_any(
        decode_profile,
        (
            "npulightningindexerhicached",
            "lightningindexerhicached",
            "npuquantlightningindexer",
            "quantlightningindexer",
        ),
        f"{decode_name} predicted LightningIndexer",
    )

if io_backend == "kvio":
    require_any(
        decode_profile,
        ("npugetputbatch", "getputbatch"),
        f"{decode_name} KVIO GET",
    )
    if scenario == "pd":
        prefill_profile = read_profile("Prefill")
        require_any(
            prefill_profile,
            ("npugetputbatch", "getputbatch"),
            "Prefill KVIO PUT",
        )
elif io_backend == "kvgather_sim":
    require_any(
        decode_profile,
        ("asukvgather", "aclnnasukvgather"),
        f"{decode_name} ASU KV Gather",
    )

print(
    "PASS: profiler contains DSA lookup/update and Sparse Flash Attention"
    + (
        " plus grouped hidden-state prefetch"
        if prefetch_enabled
        else ""
    )
    + (
        " plus KVIO PUT/GET"
        if io_backend == "kvio"
        else " plus ASU KV Gather"
        if io_backend == "kvgather_sim"
        else ""
    )
)
PY
fi

echo
echo "PASS: DSA Offload $SCENARIO scenario completed with connector=$CONNECTOR and io_backend=$IO_BACKEND."
if [[ "$ENABLE_PREFETCH_WITH_HIDDEN_STATES" == "1" ]]; then
    if [[ "$VERIFY_PATH" == "1" ]]; then
        echo "VALIDATED: grouped hidden-state prefetch lookup and Indexer execution with prefetch_top_k=$PREFETCH_TOP_K."
    else
        echo "EXERCISED: grouped hidden-state prefetch with prefetch_top_k=$PREFETCH_TOP_K."
        echo "STILL NEEDED: rerun with --verify-path for prefetch operator evidence."
    fi
fi
if [[ "$IO_BACKEND" == "mock" ]]; then
    echo "VALIDATED: config/bootstrap, request lifecycle, lookup, Hot Cache/SFA path, and role-specific control flow."
    echo "NOT VALIDATED: capacity-layer full-block PUT/token GET or output accuracy; rerun with --io-backend kvio."
elif [[ "$IO_BACKEND" == "kvgather_sim" ]]; then
    echo "VALIDATED: config/bootstrap, request lifecycle, lookup, Hot Cache/SFA path, and ASU KV Gather execution."
    echo "NOT VALIDATED: external payload correctness; kvgather_sim uses synthetic zero source blocks."
else
    echo "VALIDATED: config/bootstrap, request lifecycle, lookup, Hot Cache/SFA path, and KVIO PUT/GET execution."
    echo "STILL NEEDED: compare output tokens with a known-good non-offload baseline for accuracy sign-off."
fi
echo "Focused diagnostics:"
if [[ "$SCENARIO" == "pd" ]]; then
    echo "  grep -Ein 'dsa_offload|mooncake|localshm|npu_get_put|error|traceback' '$PREFILL_LOG' '$DECODE_LOG' '$PROXY_LOG'"
else
    echo "  grep -Ein 'dsa_offload|mooncake|localshm|npu_get_put|error|traceback' '$BOTH_LOG'"
fi
