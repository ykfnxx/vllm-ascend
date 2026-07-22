#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
SOURCE_MODEL="${SOURCE_MODEL:-/models/GLM-5.1-w4a8}"
REDUCED_ROOT="${REDUCED_MODELS_CONTAINER_PATH:-/models-reduced}"
MODEL_RUNTIME="${DMP_MODEL_RUNTIME_PYTHON_PATH:-$SCRIPT_DIR/dmp-model-runtime/python}"
BASE_ASCEND_ROOT="/vllm-workspace/vllm-ascend/vllm_ascend"
BASE_OP_API="$BASE_ASCEND_ROOT/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so"
BASE_EXTENSION="$(find "$BASE_ASCEND_ROOT" -maxdepth 1 -type f \
    -name 'vllm_ascend_C*.so' -print -quit 2>/dev/null || true)"
BASE_KERNELS="$(find "$BASE_ASCEND_ROOT" -maxdepth 1 -type f \
    -name '*vllm_ascend_kernels*.so' -print -quit 2>/dev/null || true)"
STATIC_CHECK_ONLY="${DMP_A3_STATIC_CHECK_ONLY:-0}"
declare -a errors=()

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi

if [[ "$VISIBLE_DEVICES" == *,* ]]; then
    errors+=("VISIBLE_DEVICES must select exactly one physical NPU, got: $VISIBLE_DEVICES")
fi

for command_name in python3 pip3 cmake make c++ nm sha256sum split; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        errors+=("required command is missing: $command_name")
    fi
done

required_files=(
    "$BASE_ASCEND_ROOT/_build_info.py"
    "$BASE_OP_API"
    "$SOURCE_MODEL/config.json"
    "$SCRIPT_DIR/resolve_model_runtime_wheels.sh"
    "$SCRIPT_DIR/pip-cache-dual-attention/op/ascendc/CMakeLists.txt"
    "$SCRIPT_DIR/dmp-lookup-maintain/build_and_install.sh"
)
for required_file in "${required_files[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        errors+=("required file is missing: $required_file")
    fi
done

if [[ -z "$BASE_EXTENSION" ]]; then
    errors+=("A3 image extension is missing after the source mount: vllm_ascend_C*.so")
fi
if [[ -z "$BASE_KERNELS" ]]; then
    errors+=("A3 image kernel library is missing after the source mount: vllm_ascend_kernels*.so")
fi

if [[ -f "$BASE_ASCEND_ROOT/_build_info.py" ]] && \
   ! grep -q '__device_type__.*A3' "$BASE_ASCEND_ROOT/_build_info.py"; then
    errors+=("vllm-ascend _build_info.py is not marked for A3")
fi

if [[ ! -d "$REDUCED_ROOT" || ! -w "$REDUCED_ROOT" ]]; then
    errors+=("reduced-model directory is missing or not writable: $REDUCED_ROOT")
fi

model_runtime_ready=0
if PYTHONPATH="$MODEL_RUNTIME${PYTHONPATH:+:$PYTHONPATH}" python3 -c '
from transformers import AutoConfig
AutoConfig.for_model("glm_moe_dsa")
' >/dev/null 2>&1; then
    model_runtime_ready=1
fi

source "$SCRIPT_DIR/resolve_model_runtime_wheels.sh"
dmp_resolve_model_runtime_wheels
if [[ "$model_runtime_ready" != "1" ]]; then
    if [[ -z "$DMP_RESOLVED_TRANSFORMERS_WHEEL" || \
          ! -f "$DMP_RESOLVED_TRANSFORMERS_WHEEL" ]]; then
        errors+=("transformers-5.2.0*.whl is missing under /dmp-host")
    fi
    if [[ -z "$DMP_RESOLVED_HUGGINGFACE_HUB_WHEEL" || \
          ! -f "$DMP_RESOLVED_HUGGINGFACE_HUB_WHEEL" ]]; then
        errors+=("huggingface_hub-1.22.0*.whl is missing under /dmp-host")
    fi
fi

export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
native_ops_info="not probed (fast startup)"
if [[ "$STATIC_CHECK_ONLY" != "1" && -n "$BASE_EXTENSION" && \
      -n "$BASE_KERNELS" && -f "$BASE_OP_API" ]]; then
    if ! native_ops_info="$(
        PYTHONPATH="/vllm-workspace/vllm-ascend${PYTHONPATH:+:$PYTHONPATH}" \
            python3 - <<'PY' 2>&1
import importlib

import torch
import torch_npu  # noqa: F401

module = importlib.import_module("vllm_ascend.vllm_ascend_C")
required = (
    "moe_gating_top_k",
    "npu_moe_init_routing_custom",
    "npu_lightning_indexer_quant",
    "npu_sparse_flash_attention",
    "npu_add_rms_norm_bias",
)
missing = [name for name in required if not hasattr(torch.ops._C_ascend, name)]
if missing:
    raise RuntimeError(f"missing _C_ascend operators: {missing}")
print(module.__file__)
print(", ".join(required))
PY
    )"; then
        errors+=("A3 base extension cannot register its PyTorch operators: $native_ops_info")
    fi
fi

device_info="not probed (fast startup)"
if [[ "$STATIC_CHECK_ONLY" != "1" ]] && command -v python3 >/dev/null 2>&1; then
    if ! device_info="$(python3 - <<'PY' 2>&1
import torch_npu

assert torch_npu.npu.device_count() >= 1, "no visible NPU"
print(torch_npu.npu.get_device_name(0))
print(torch_npu.npu.get_soc_version())
PY
    )"; then
        errors+=("torch_npu cannot access logical NPU 0 through VISIBLE_DEVICES=$VISIBLE_DEVICES: $device_info")
    fi
fi

if (( ${#errors[@]} > 0 )); then
    echo "A3 single-card prerequisite check found ${#errors[@]} problem(s):" >&2
    for error in "${errors[@]}"; do
        echo "  - $error" >&2
    done
    exit 1
fi

free_kb="$(df -Pk "$REDUCED_ROOT" | awk 'NR == 2 {print $4}')"
echo "A3_SINGLE_CARD_PREREQS_OK"
echo "  physical device selection: $VISIBLE_DEVICES"
echo "  logical device: ${device_info//$'\n'/, }"
echo "  source model: $SOURCE_MODEL"
echo "  reduced-model free space: ${free_kb} KiB"
echo "  A3 native extension: ${native_ops_info%%$'\n'*}"
if [[ "$model_runtime_ready" == "1" ]]; then
    echo "  GLM Transformers runtime: already persistent or bundled"
else
    echo "  Transformers wheel: $DMP_RESOLVED_TRANSFORMERS_WHEEL"
    echo "  Hugging Face Hub wheel: $DMP_RESOLVED_HUGGINGFACE_HUB_WHEEL"
fi
