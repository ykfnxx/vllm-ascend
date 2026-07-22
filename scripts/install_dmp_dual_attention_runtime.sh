#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_ROOT="${DMP_DUAL_ATTENTION_OP_ROOT:-$SCRIPT_DIR/pip-cache-dual-attention}"
RUNTIME_ROOT="${DMP_DUAL_ATTENTION_RUNTIME_ROOT:-$SCRIPT_DIR/dmp-runtime}"
PERSISTENT_OPP="$RUNTIME_ROOT/opp"
PERSISTENT_PYTHON="$RUNTIME_ROOT/python"
MODEL_RUNTIME_PYTHON="${DMP_MODEL_RUNTIME_PYTHON_PATH:-$SCRIPT_DIR/dmp-model-runtime/python}"
INSTALLED_OP_API="$PERSISTENT_OPP/vendors/customize/op_api/lib/libcust_opapi.so"
OPP_STAMP="$PERSISTENT_OPP/.dmp-opp-package.sha256"
WHEEL_STAMP="$PERSISTENT_PYTHON/.dmp-custom-ops-wheel.sha256"
source "$SCRIPT_DIR/resolve_model_runtime_wheels.sh"
dmp_resolve_model_runtime_wheels
TRANSFORMERS_WHEEL="$DMP_RESOLVED_TRANSFORMERS_WHEEL"
HUGGINGFACE_HUB_WHEEL="$DMP_RESOLVED_HUGGINGFACE_HUB_WHEEL"
MODEL_RUNTIME_STAMP="$MODEL_RUNTIME_PYTHON/.dmp-model-runtime.sha256"

mkdir -p "$PERSISTENT_OPP" "$PERSISTENT_PYTHON" "$MODEL_RUNTIME_PYTHON"

RUN_PKG=""
if [[ -d "$OP_ROOT/op/ascendc/output" ]]; then
    RUN_PKG="$(find "$OP_ROOT/op/ascendc/output" -maxdepth 1 -type f \
        -name 'CANN-custom_ops-*.run' -print | LC_ALL=C sort | tail -1)"
fi
if [[ -z "$RUN_PKG" && ! -f "$INSTALLED_OP_API" ]]; then
    echo "Cached OPP installer not found. Run build_dmp_dual_attention_ops.sh once." >&2
    exit 1
fi

WHEEL=""
if [[ -d "$OP_ROOT/op/torch_ops_extension/dist" ]]; then
    WHEEL="$(find "$OP_ROOT/op/torch_ops_extension/dist" -maxdepth 1 -type f \
        -name '*.whl' -print | LC_ALL=C sort | tail -1)"
fi
expected_opp_stamp=""
expected_wheel_stamp=""
if [[ -n "$RUN_PKG" ]]; then
    expected_opp_stamp="$(sha256sum "$RUN_PKG" | awk '{print $1}')"
fi
if [[ -n "$WHEEL" ]]; then
    expected_wheel_stamp="$(sha256sum "$WHEEL" | awk '{print $1}')"
fi
installed_opp_stamp=""
installed_wheel_stamp=""
if [[ -f "$OPP_STAMP" ]]; then
    installed_opp_stamp="$(<"$OPP_STAMP")"
fi
if [[ -f "$WHEEL_STAMP" ]]; then
    installed_wheel_stamp="$(<"$WHEEL_STAMP")"
fi

have_model_wheels=1
for wheel in "$TRANSFORMERS_WHEEL" "$HUGGINGFACE_HUB_WHEEL"; do
    [[ -n "$wheel" && -f "$wheel" ]] || have_model_wheels=0
done
expected_model_runtime_stamp=""
if [[ "$have_model_wheels" == "1" ]]; then
    expected_model_runtime_stamp="$(
        sha256sum "$TRANSFORMERS_WHEEL" "$HUGGINGFACE_HUB_WHEEL" \
            | awk '{print $1}' \
            | sha256sum \
            | awk '{print $1}'
    )"
fi
installed_model_runtime_stamp=""
if [[ -f "$MODEL_RUNTIME_STAMP" ]]; then
    installed_model_runtime_stamp="$(<"$MODEL_RUNTIME_STAMP")"
fi
model_runtime_is_ready=0
if PYTHONPATH="$MODEL_RUNTIME_PYTHON${PYTHONPATH:+:$PYTHONPATH}" python3 -c '
from transformers import AutoConfig
AutoConfig.for_model("glm_moe_dsa")
' >/dev/null 2>&1; then
    model_runtime_is_ready=1
fi

if [[ "$have_model_wheels" == "1" && \
      ( "$installed_model_runtime_stamp" != "$expected_model_runtime_stamp" || \
        "$model_runtime_is_ready" != "1" ) ]]; then
    echo "Deploying Transformers model runtime to persistent storage..."
    pip3 install \
        "$HUGGINGFACE_HUB_WHEEL" \
        "$TRANSFORMERS_WHEEL" \
        --target "$MODEL_RUNTIME_PYTHON" \
        --upgrade \
        --no-deps
    printf '%s\n' "$expected_model_runtime_stamp" > "$MODEL_RUNTIME_STAMP"
fi

if ! PYTHONPATH="$MODEL_RUNTIME_PYTHON:$PERSISTENT_PYTHON${PYTHONPATH:+:$PYTHONPATH}" python3 -c '
from transformers import AutoConfig
AutoConfig.for_model("glm_moe_dsa")
' >/dev/null 2>&1; then
    echo "The A3 image Transformers runtime does not support model_type=glm_moe_dsa." >&2
    echo "Provide Transformers 5.2 wheels or use the official v0.18.0 A3 image." >&2
    exit 1
fi
if [[ "$have_model_wheels" != "1" ]]; then
    echo "Using the GLM-5.1 Transformers runtime bundled in the A3 image."
fi

if [[ -n "$RUN_PKG" && ( ! -f "$INSTALLED_OP_API" || "$installed_opp_stamp" != "$expected_opp_stamp" ) ]]; then
    echo "Deploying cached DMP Dual-Attention OPP to persistent storage..."
    rm -rf "$PERSISTENT_OPP/vendors/customize"
    chmod +x "$RUN_PKG"
    "$RUN_PKG" --quiet --install-path="$PERSISTENT_OPP"
    printf '%s\n' "$expected_opp_stamp" > "$OPP_STAMP"
fi

wheel_is_current=0
wheel_stamp_matches=0
if [[ -z "$WHEEL" || "$installed_wheel_stamp" == "$expected_wheel_stamp" ]]; then
    wheel_stamp_matches=1
fi
if [[ "$wheel_stamp_matches" == "1" ]] && \
    PYTHONPATH="$PERSISTENT_PYTHON${PYTHONPATH:+:$PYTHONPATH}" python3 -c '
import custom_ops  # noqa: F401
import torch
required = ("npu_kv_select_out", "npu_kv_gather_out", "npu_dmp_sparse_flash_attention", "npu_da_attention_merge")
raise SystemExit(0 if all(hasattr(torch.ops.custom, name) for name in required) else 1)
' >/dev/null 2>&1; then
    wheel_is_current=1
fi

if [[ "$wheel_is_current" != "1" ]]; then
    if [[ -z "$WHEEL" ]]; then
        echo "Persistent custom_ops is missing or stale, and no cached wheel is available." >&2
        echo "Run build_dmp_dual_attention_ops.sh once." >&2
        exit 1
    fi
    echo "Deploying cached DMP Dual-Attention wheel to persistent storage..."
    find "$PERSISTENT_PYTHON" -maxdepth 1 \
        \( -name custom_ops -o -name 'custom_ops-*.dist-info' \) \
        -exec rm -rf {} +
    pip3 install "$WHEEL" --target "$PERSISTENT_PYTHON" --upgrade --no-deps
    printf '%s\n' "$expected_wheel_stamp" > "$WHEEL_STAMP"
fi

for symbol in aclnnKVSelect aclnnDmpSparseFlashAttention; do
    if ! nm -D "$INSTALLED_OP_API" | grep " ${symbol}$" >/dev/null; then
        echo "Installed DMP operator library does not contain ${symbol}: $INSTALLED_OP_API" >&2
        exit 1
    fi
done

echo "DMP Dual-Attention persistent runtime is ready: $RUNTIME_ROOT"
echo "OPP SHA256:   ${expected_opp_stamp:-$installed_opp_stamp}"
echo "Wheel SHA256: ${expected_wheel_stamp:-$installed_wheel_stamp}"
