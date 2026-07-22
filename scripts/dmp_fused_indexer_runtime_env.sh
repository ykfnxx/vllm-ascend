#!/usr/bin/env bash

# Runtime settings for scheme 3: fused Indexer+Select request pool on S0,
# one local-HBM mock KVIO per microbatch on S1, then selected-cache SFA on S0.
_dmp_fused_nounset_was_on=0
case "$-" in
    *u*) _dmp_fused_nounset_was_on=1 ;;
esac
set +u

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

_dmp_fused_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VISIBLE_DEVICES="${VISIBLE_DEVICES:-5}"
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export VLLM_ASCEND_ENABLE_DMP=1
export VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT="${VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT:-1}"
export VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN=0
export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION="${VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION:-0}"

if [[ "$VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT" == "1" && \
      "$VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION" == "1" ]]; then
    echo "Fused Indexer+KVSelect and DMP Dual-Attention cannot both be enabled." >&2
    return 1 2>/dev/null || exit 1
fi

export DMP_FUSED_INDEXER_ROOT="${DMP_FUSED_INDEXER_ROOT:-$_dmp_fused_script_dir/dmp-fused-indexer-kv-select}"
_dmp_fused_vendor="$DMP_FUSED_INDEXER_ROOT/opp/vendors/customize"
_dmp_fused_python="$DMP_FUSED_INDEXER_ROOT/torch_extension"
_dmp_fused_lookup_vendor="${DMP_LOOKUP_MAINTAIN_OPP_PATH:-$_dmp_fused_script_dir/dmp-lookup-maintain/opp/vendors/customize}"
_dmp_fused_lookup_aicpu="$_dmp_fused_lookup_vendor/op_impl/aicpu_transformer"
_dmp_fused_lookup_python="${DMP_LOOKUP_MAINTAIN_PYTHON_PATH:-$_dmp_fused_script_dir/dmp-lookup-maintain/torch_extension}"
_dmp_fused_dual_vendor="${DMP_DUAL_ATTENTION_OPP_PATH:-$_dmp_fused_script_dir/dmp-runtime/opp/vendors/customize}"
_dmp_fused_dual_python="${DMP_DUAL_ATTENTION_PYTHON_PATH:-$_dmp_fused_script_dir/dmp-runtime/python}"
export DMP_FUSED_INDEXER_OPP_PATH="$_dmp_fused_vendor"
export DMP_FUSED_INDEXER_PYTHON_PATH="$_dmp_fused_python"

if [[ "$VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT" == "1" ]]; then
    if [[ ! -f "$_dmp_fused_vendor/op_api/lib/libcust_opapi.so" ]]; then
        echo "Fused Indexer+KVSelect OPP is missing: $_dmp_fused_vendor" >&2
        echo "Run /workspace/scripts/build_dmp_fused_indexer_kv_select.sh first." >&2
        return 1 2>/dev/null || exit 1
    fi
    if ! compgen -G "$_dmp_fused_python/lightning_indexer_decode_custom_ops/*.so" >/dev/null; then
        echo "Fused Indexer+KVSelect torch extension is missing: $_dmp_fused_python" >&2
        echo "Run /workspace/scripts/build_dmp_fused_indexer_kv_select.sh first." >&2
        return 1 2>/dev/null || exit 1
    fi
    if [[ ! -f "$_dmp_fused_lookup_vendor/op_api/lib/libcust_opapi.so" ]] || \
       ! compgen -G "$_dmp_fused_lookup_python/dmp_lookup_maintain_custom_ops/*.so" >/dev/null; then
        echo "Scheme-3 KVIO runtime is missing: $_dmp_fused_lookup_vendor" >&2
        echo "Run /workspace/scripts/build_a3_dmp_runtime.sh first." >&2
        return 1 2>/dev/null || exit 1
    fi
    if [[ ! -f "$_dmp_fused_dual_vendor/op_api/lib/libcust_opapi.so" ]] || \
       ! compgen -G "$_dmp_fused_dual_python/custom_ops/*.so" >/dev/null; then
        echo "Scheme-3 SFA runtime is missing: $_dmp_fused_dual_vendor" >&2
        echo "Run /workspace/scripts/build_a3_dmp_runtime.sh first." >&2
        return 1 2>/dev/null || exit 1
    fi
    _dmp_fused_filtered_python=""
    IFS=':' read -r -a _dmp_fused_python_entries <<< "${PYTHONPATH:-}"
    for _dmp_fused_entry in "${_dmp_fused_python_entries[@]}"; do
        [[ -n "$_dmp_fused_entry" ]] || continue
        case "${_dmp_fused_entry%/}" in
            "${_dmp_fused_python%/}"|"${_dmp_fused_lookup_python%/}"|"${_dmp_fused_dual_python%/}")
                continue
                ;;
        esac
        _dmp_fused_filtered_python="${_dmp_fused_filtered_python:+${_dmp_fused_filtered_python}:}${_dmp_fused_entry}"
    done
    export PYTHONPATH="$_dmp_fused_python:$_dmp_fused_lookup_python:$_dmp_fused_dual_python${_dmp_fused_filtered_python:+:${_dmp_fused_filtered_python}}"

    _dmp_fused_filtered_opp=""
    IFS=':' read -r -a _dmp_fused_opp_entries <<< "${ASCEND_CUSTOM_OPP_PATH:-}"
    for _dmp_fused_entry in "${_dmp_fused_opp_entries[@]}"; do
        [[ -n "$_dmp_fused_entry" ]] || continue
        case "${_dmp_fused_entry%/}" in
            "${_dmp_fused_vendor%/}"|"${_dmp_fused_lookup_vendor%/}"|"${_dmp_fused_lookup_aicpu%/}"|"${_dmp_fused_dual_vendor%/}")
                continue
                ;;
        esac
        _dmp_fused_filtered_opp="${_dmp_fused_filtered_opp:+${_dmp_fused_filtered_opp}:}${_dmp_fused_entry}"
    done
    export ASCEND_CUSTOM_OPP_PATH="$_dmp_fused_vendor:$_dmp_fused_lookup_vendor:$_dmp_fused_dual_vendor${_dmp_fused_filtered_opp:+:${_dmp_fused_filtered_opp}}"
fi

if [[ "$_dmp_fused_nounset_was_on" == "1" ]]; then
    set -u
fi
unset _dmp_fused_nounset_was_on _dmp_fused_script_dir _dmp_fused_vendor \
    _dmp_fused_python _dmp_fused_lookup_vendor _dmp_fused_lookup_aicpu \
    _dmp_fused_lookup_python \
    _dmp_fused_dual_vendor _dmp_fused_dual_python _dmp_fused_filtered_opp \
    _dmp_fused_opp_entries _dmp_fused_filtered_python \
    _dmp_fused_python_entries _dmp_fused_entry
