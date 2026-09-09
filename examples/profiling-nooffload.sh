#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_DIR="$(cd -- "${SCRIPT_DIR}/../../vllm-ascend" && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/model/GLM-5.2-w4a4c8-mxfp4-l10}"
PREFETCH_MODE="${PREFETCH_MODE:-offload}"
PREFETCH_TOP_K="${PREFETCH_TOP_K:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

usage() {
    echo "Usage: $0 [offload|prefetch]" >&2
    echo "Both modes enable offload; prefetch additionally enables hidden-state prefetch." >&2
    echo "MODEL_PATH selects the model checkpoint; default: /data/model/GLM-5.2-w4a4c8-mxfp4-l10." >&2
    echo "PREFETCH_MODE may also be set through the environment; default: offload." >&2
    echo "PREFETCH_TOP_K controls the predicted Top-K width; default: 2048." >&2
    echo "GPU_MEMORY_UTILIZATION controls the per-engine memory fraction; default: 0.90." >&2
}

case "$#" in
    0) ;;
    1)
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            *) PREFETCH_MODE="$1" ;;
        esac
        ;;
    *)
        usage
        exit 2
        ;;
esac

case "${PREFETCH_MODE}" in
    offload)
        RUN_MODE="offload"
        PREFETCH_ENABLED="false"
        PREFETCH_ARGS=()
        ;;
    prefetch)
        RUN_MODE="prefetch"
        PREFETCH_ENABLED="true"
        PREFETCH_ARGS=(
            --enable-prefetch-with-hidden-states
            --prefetch-top-k "${PREFETCH_TOP_K}"
        )
        ;;
    *)
        usage
        exit 2
        ;;
esac

RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/profiling-w4/no-offload-bs4-${RUN_ID}}"
mkdir -p "${OUTPUT_DIR}"

unset http_proxy
unset https_proxy

echo "Profiling offload=enabled, hidden_state_prefetch=${PREFETCH_ENABLED}, prefetch_top_k=${PREFETCH_TOP_K}"
echo "Decode mode: io_backend=kvgather_sim, cudagraph_mode=FULL_DECODE_ONLY, mtp_speculative_tokens=3"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"

#cd "${VLLM_ASCEND_DIR}"
VLLM_ASCEND_ENABLE_MLAPO=0 \
bash dsa_offload_probe_hhm.sh \
    --model "${MODEL_PATH}" \
    --scenario pd \
    --connector local-shm \
    --cudagraph-mode FULL_DECODE_ONLY \
    --host-ip 90.90.93.29 \
    --prefill-device 3 \
    --decode-device 2 \
    --max-tokens 8 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --ifname ens6f1 \
    --batch-size 4 \
    --max-model-len 68000 \
    --no-dsa-offload \
    --profile \
    --input-jsonl longbench_narrativeqa_64k.jsonl \
    --log-dir "${OUTPUT_DIR}" \
    "${PREFETCH_ARGS[@]}"

# --mtp-speculative-tokens 3 \