#!/usr/bin/env bash
set -euo pipefail

IMAGE="${DMP_A3_IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0-a3-openeuler}"
MODEL="${MODEL_HOST_PATH:-/mnt/models/GLM-5.1-w4a8}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
REDUCED_LAYERS="${REDUCED_LAYERS:-auto-through-first-MoE}"
PREFIX="${DMP_ROOT:-/root/dmp}"
REDUCED_ROOT="${REDUCED_MODELS_HOST_PATH:-$PREFIX/reduced-models}"

docker image inspect "$IMAGE" >/dev/null
[[ -d "$MODEL" ]] || { echo "Model directory is missing: $MODEL" >&2; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "config.json is missing: $MODEL" >&2; exit 1; }

npu_count="$(find /dev -maxdepth 1 -type c -name 'davinci[0-9]*' 2>/dev/null | wc -l)"
weight_bytes="$(find "$MODEL" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.bin' \) -printf '%s\n' \
    | awk '{sum += $1} END {print sum + 0}')"

echo "A3 image:       $IMAGE"
echo "NPU devices:    $npu_count"
echo "Model:          $MODEL"
echo "Weight bytes:   $weight_bytes"
echo "Requested TP:   $TP_SIZE"
echo "Reduced layers: $REDUCED_LAYERS"
echo "Reduced output: $REDUCED_ROOT"

if (( npu_count < TP_SIZE )); then
    echo "Not enough visible host NPUs for TP=$TP_SIZE." >&2
    exit 1
fi
if (( TP_SIZE == 1 && weight_bytes > 59000000000 )); then
    echo "Full weights exceed one card; the container will create a reduced checkpoint (${REDUCED_LAYERS})."
fi

mkdir -p "$REDUCED_ROOT"
echo "A3 single-card host preflight OK."
