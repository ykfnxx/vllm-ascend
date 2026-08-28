#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTAINER_NAME="${A5_CONTAINER_NAME:-vllm-ascend-a5-kvgather-sim}"
PREFILL_DEVICE="${A5_PREFILL_DEVICE:-3}"
DECODE_DEVICE="${A5_DECODE_DEVICE:-5}"
HOST_IP="${A5_PD_HOST_IP:-90.90.93.29}"
IFNAME="${A5_PD_IFNAME:-ens6f1}"
MODEL_PATH="${MODEL_PATH:-/home/y00852214/repos/glm-moe-dsa}"
STAMP="${A5_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME=a5_dsa_offload_mtp_kvgathersim_64x128k_graph
RUN_DIR="${A5_RESULTS_ROOT:-$REPO_ROOT/scripts/results}/$STAMP"
LOG_DIR="$RUN_DIR/logs"
PROFILE_DIR="$RUN_DIR/profile/${STAMP}_${RUN_NAME}"
FULL_LOG="$LOG_DIR/${STAMP}_${RUN_NAME}.full.log"
RESULT_FILE="$LOG_DIR/${STAMP}_${RUN_NAME}_result.txt"

source_commit() {
    if [[ -r "$REPO_ROOT/SNAPSHOT_COMMIT" ]]; then
        cat "$REPO_ROOT/SNAPSHOT_COMMIT"
    else
        git -C "$REPO_ROOT" rev-parse HEAD
    fi
}

[[ "$PREFILL_DEVICE" != "$DECODE_DEVICE" ]] || {
    echo "A5_PREFILL_DEVICE and A5_DECODE_DEVICE must differ." >&2
    exit 2
}
mkdir -p "$LOG_DIR" "$PROFILE_DIR"

echo "A5 DSA Offload MTP KV Gather Sim Decode Graph Profile"
echo "  source commit: $(source_commit)"
echo "  model:         $MODEL_PATH"
echo "  physical NPUs: Prefill=$PREFILL_DEVICE Decode=$DECODE_DEVICE"
echo "  workload:      batch=64 prompt=131072 output=10"
echo "  lookup:        dsa_offload_lookup_update_batch"
echo "  gather:        real asu_kv_gather sim kernel, synthetic source payload"
echo "  execution:     eager Prefill + MTP FULL_DECODE_ONLY Graph Decode"
echo "  MLAPO:         disabled (native SFA preprocessing for A5 graph capture)"
echo "  DMP:           disabled"
echo "  profile:       Decode only"
echo "  results:       $RUN_DIR"

A5_CONTAINER_NAME="$CONTAINER_NAME" \
A5_VISIBLE_DEVICES="$PREFILL_DEVICE,$DECODE_DEVICE" \
A5_REPO_ROOT="$REPO_ROOT" \
MODEL_PATH="$MODEL_PATH" \
bash "$SCRIPT_DIR/start_a5_dsa_offload_container.sh"

docker exec -i \
    -e "A5_DECODE_DEVICE=$DECODE_DEVICE" \
    -e "A5_RUN_TIMESTAMP=$STAMP" \
    -e "A5_LOG_ROOT_IN_CONTAINER=/vllm-workspace/vllm-ascend/scripts/results/$STAMP/logs" \
    -e "A5_FORCE_REBUILD=${A5_FORCE_REBUILD:-0}" \
    -e "MAX_JOBS=${MAX_JOBS:-8}" \
    "$CONTAINER_NAME" \
    bash /vllm-workspace/vllm-ascend/scripts/prepare_a5_dsa_offload_mtp_kvgathersim_runtime.sh

set +e
docker exec -i \
    -e "A5_PREFILL_DEVICE=$PREFILL_DEVICE" \
    -e "A5_DECODE_DEVICE=$DECODE_DEVICE" \
    -e "A5_PD_HOST_IP=$HOST_IP" \
    -e "A5_PD_IFNAME=$IFNAME" \
    -e "A5_SERVICE_LOG_DIR=/vllm-workspace/vllm-ascend/scripts/results/$STAMP/logs/services" \
    -e "A5_PROFILE_DIR=/vllm-workspace/vllm-ascend/scripts/results/$STAMP/profile/${STAMP}_${RUN_NAME}" \
    "$CONTAINER_NAME" \
    bash /vllm-workspace/vllm-ascend/examples/dsa_offload_mtp_kvgathersim_graph_profile.sh \
    2>&1 | tee "$FULL_LOG"
status=${PIPESTATUS[0]}
set -e

if ((status != 0)); then
    {
        echo "A5_DSA_OFFLOAD_MTP_KVGATHER_SIM_GRAPH_FAILED: status=$status"
        echo "full_log=$FULL_LOG"
        echo "service_logs=$LOG_DIR/services"
    } | tee "$RESULT_FILE" >&2
    exit "$status"
fi

trace_dir="$(find "$PROFILE_DIR" -type d -name '*_ascend_pt' -print -quit)"
[[ -n "$trace_dir" ]] || {
    echo "Decode profile trace is missing under $PROFILE_DIR" >&2
    exit 1
}
{
    echo "A5_DSA_OFFLOAD_MTP_KVGATHER_SIM_64X128K_GRAPH_PASSED"
    echo "source_commit=$(source_commit)"
    echo "workload=batch64_prompt131072_output10"
    echo "mtp_speculative_tokens=1"
    echo "prefill=eager_npu$PREFILL_DEVICE"
    echo "decode=full_decode_only_graph_npu$DECODE_DEVICE"
    echo "lookup=dsa_offload_lookup_update_batch"
    echo "gather=asu_kv_gather_sim"
    echo "gather_source_payload=synthetic_zero"
    echo "mlapo=disabled"
    echo "dmp=disabled"
    echo "profile_scope=decode_only"
    echo "mindstudio_import=$trace_dir"
    echo "result_dir=$RUN_DIR"
} | tee "$RESULT_FILE"
