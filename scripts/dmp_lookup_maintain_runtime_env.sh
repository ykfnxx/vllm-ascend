#!/usr/bin/env bash

_dmp_lookup_nounset_was_on=0
case "$-" in
    *u*) _dmp_lookup_nounset_was_on=1 ;;
esac
set +u

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

_dmp_lookup_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_dmp_lookup_script_dir/resolve_model_runtime_wheels.sh"
export VISIBLE_DEVICES="${VISIBLE_DEVICES:-6}"
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export VLLM_ASCEND_ENABLE_DMP=1
export VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN="${VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN:-1}"
export VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT="${VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT:-0}"
export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION="${VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION:-0}"

_dmp_lookup_mode_count=$((
    VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN +
    VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT +
    VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION
))
if (( _dmp_lookup_mode_count > 1 )); then
    echo "DMP Lookup/Maintain, Fused Indexer, and Dual-Attention are mutually exclusive." >&2
    return 1 2>/dev/null || exit 1
fi

export DMP_LOOKUP_MAINTAIN_ROOT="${DMP_LOOKUP_MAINTAIN_ROOT:-$_dmp_lookup_script_dir/dmp-lookup-maintain}"
export DMP_LOOKUP_MAINTAIN_INSTALL_OPP_PATH="${DMP_LOOKUP_MAINTAIN_INSTALL_OPP_PATH:-$DMP_LOOKUP_MAINTAIN_ROOT/opp}"
_dmp_lookup_vendor="$DMP_LOOKUP_MAINTAIN_INSTALL_OPP_PATH/vendors/customize"
_dmp_lookup_aicpu_repo="$_dmp_lookup_vendor/op_impl/aicpu_transformer"
_dmp_lookup_python="$DMP_LOOKUP_MAINTAIN_ROOT/torch_extension"
export DMP_LOOKUP_MAINTAIN_OPP_PATH="$_dmp_lookup_vendor"
export DMP_LOOKUP_MAINTAIN_AICPU_OPP_PATH="$_dmp_lookup_aicpu_repo"
export DMP_LOOKUP_MAINTAIN_PYTHON_PATH="$_dmp_lookup_python"
export DMP_DUAL_ATTENTION_RUNTIME_ROOT="${DMP_DUAL_ATTENTION_RUNTIME_ROOT:-$_dmp_lookup_script_dir/dmp-runtime}"
export DMP_DUAL_ATTENTION_OPP_PATH="${DMP_DUAL_ATTENTION_OPP_PATH:-$DMP_DUAL_ATTENTION_RUNTIME_ROOT/opp/vendors/customize}"
_dmp_lookup_dual_python="$DMP_DUAL_ATTENTION_RUNTIME_ROOT/python"
_dmp_model_runtime="${DMP_MODEL_RUNTIME_PYTHON_PATH:-$_dmp_lookup_script_dir/dmp-model-runtime/python}"

# Keep the GLM-5 Transformers runtime under the host-mounted scripts tree.
if ! PYTHONPATH="$_dmp_model_runtime${PYTHONPATH:+:$PYTHONPATH}" python3 -c '
from transformers import AutoConfig
AutoConfig.for_model("glm_moe_dsa")
' >/dev/null 2>&1; then
    dmp_resolve_model_runtime_wheels
    _dmp_transformers_wheel="$DMP_RESOLVED_TRANSFORMERS_WHEEL"
    _dmp_huggingface_wheel="$DMP_RESOLVED_HUGGINGFACE_HUB_WHEEL"
    for _dmp_model_wheel in "$_dmp_huggingface_wheel" "$_dmp_transformers_wheel"; do
        if [[ -z "$_dmp_model_wheel" || ! -f "$_dmp_model_wheel" ]]; then
            echo "Required GLM-5 model runtime wheel was not found." >&2
            echo "Expected huggingface_hub-1.22.0*.whl and transformers-5.2.0*.whl under /root/dmp on the host." >&2
            echo "The new container exposes them under /dmp-host." >&2
            return 1 2>/dev/null || exit 1
        fi
    done
    mkdir -p "$_dmp_model_runtime"
    python3 -m pip install \
        "$_dmp_huggingface_wheel" \
        "$_dmp_transformers_wheel" \
        --target "$_dmp_model_runtime" \
        --upgrade \
        --no-deps
fi
export PYTHONPATH="$_dmp_model_runtime${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN" == "1" ]]; then
    if [[ ! -f "$_dmp_lookup_vendor/op_api/lib/libcust_opapi.so" ]]; then
        echo "Lookup/Maintain OPP is missing: $_dmp_lookup_vendor" >&2
        echo "Run /workspace/scripts/build_dmp_lookup_maintain.sh first." >&2
        return 1 2>/dev/null || exit 1
    fi
    if [[ ! -f "$_dmp_lookup_aicpu_repo/op_impl/cpu/config/cust_aicpu_kernel.json" || \
          ! -f "$_dmp_lookup_aicpu_repo/op_impl/cpu/aicpu_kernel/impl/libtransformer_aicpu_kernels.so" ]]; then
        echo "Lookup/Maintain AICPU repository is incomplete: $_dmp_lookup_aicpu_repo" >&2
        echo "Run /workspace/scripts/build_dmp_lookup_maintain.sh first." >&2
        return 1 2>/dev/null || exit 1
    fi
    if ! compgen -G \
        "$_dmp_lookup_python/dmp_lookup_maintain_custom_ops/*.so" >/dev/null; then
        echo "Lookup/Maintain torch extension is missing: $_dmp_lookup_python" >&2
        echo "Run /workspace/scripts/build_dmp_lookup_maintain.sh first." >&2
        return 1 2>/dev/null || exit 1
    fi
    if [[ ! -f "$DMP_DUAL_ATTENTION_OPP_PATH/op_api/lib/libcust_opapi.so" ]]; then
        echo "Dual-Attention SFA/merge OPP is missing: $DMP_DUAL_ATTENTION_OPP_PATH" >&2
        echo "Run /workspace/scripts/build_dmp_dual_attention_ops.sh first." >&2
        return 1 2>/dev/null || exit 1
    fi
    if ! compgen -G "$_dmp_lookup_dual_python/custom_ops/*.so" >/dev/null; then
        echo "Dual-Attention torch extension is missing: $_dmp_lookup_dual_python" >&2
        echo "Run /workspace/scripts/build_dmp_dual_attention_ops.sh first." >&2
        return 1 2>/dev/null || exit 1
    fi
    export PYTHONPATH="$_dmp_lookup_python:$_dmp_lookup_dual_python${PYTHONPATH:+:$PYTHONPATH}"

    _dmp_lookup_filtered_opp=""
    IFS=':' read -r -a _dmp_lookup_opp_entries <<< "${ASCEND_CUSTOM_OPP_PATH:-}"
    for _dmp_lookup_entry in "${_dmp_lookup_opp_entries[@]}"; do
        [[ -n "$_dmp_lookup_entry" ]] || continue
        [[ "${_dmp_lookup_entry%/}" == "${_dmp_lookup_vendor%/}" ]] && continue
        [[ "${_dmp_lookup_entry%/}" == "${_dmp_lookup_aicpu_repo%/}" ]] && continue
        [[ "${_dmp_lookup_entry%/}" == "${DMP_DUAL_ATTENTION_OPP_PATH%/}" ]] && continue
        _dmp_lookup_filtered_opp="${_dmp_lookup_filtered_opp:+${_dmp_lookup_filtered_opp}:}${_dmp_lookup_entry}"
    done
    # CANN 8.5 discovers a custom AICPU kernel only when its suffixed
    # repository is listed separately and before the surrounding vendor OPP.
    export ASCEND_CUSTOM_OPP_PATH="$_dmp_lookup_aicpu_repo:$_dmp_lookup_vendor:$DMP_DUAL_ATTENTION_OPP_PATH${_dmp_lookup_filtered_opp:+:${_dmp_lookup_filtered_opp}}"
fi

if [[ "$_dmp_lookup_nounset_was_on" == "1" ]]; then
    set -u
fi
unset _dmp_lookup_nounset_was_on _dmp_lookup_script_dir \
    _dmp_lookup_mode_count _dmp_lookup_vendor _dmp_lookup_aicpu_repo \
    _dmp_lookup_python _dmp_lookup_dual_python \
    _dmp_model_runtime _dmp_transformers_wheel _dmp_huggingface_wheel \
    _dmp_model_wheel _dmp_lookup_filtered_opp _dmp_lookup_opp_entries \
    _dmp_lookup_entry DMP_RESOLVED_TRANSFORMERS_WHEEL \
    DMP_RESOLVED_HUGGINGFACE_HUB_WHEEL
unset -f dmp_resolve_model_runtime_wheels
