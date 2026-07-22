#!/usr/bin/env bash

# Runtime settings for the standalone Indexer+KVSelect fusion. This mode does
# not enable the segmented Dual-Attention/KVGather/HIXL path.
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
export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION="${VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION:-0}"

if [[ "$VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT" == "1" && \
      "$VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION" == "1" ]]; then
    echo "Fused Indexer+KVSelect and DMP Dual-Attention cannot both be enabled." >&2
    return 1 2>/dev/null || exit 1
fi

export DMP_FUSED_INDEXER_ROOT="${DMP_FUSED_INDEXER_ROOT:-$_dmp_fused_script_dir/dmp-fused-indexer-kv-select}"
_dmp_fused_vendor="$DMP_FUSED_INDEXER_ROOT/opp/vendors/customize"
_dmp_fused_python="$DMP_FUSED_INDEXER_ROOT/torch_extension"

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
    export PYTHONPATH="$_dmp_fused_python${PYTHONPATH:+:$PYTHONPATH}"

    _dmp_fused_filtered_opp=""
    IFS=':' read -r -a _dmp_fused_opp_entries <<< "${ASCEND_CUSTOM_OPP_PATH:-}"
    for _dmp_fused_entry in "${_dmp_fused_opp_entries[@]}"; do
        [[ -n "$_dmp_fused_entry" ]] || continue
        [[ "${_dmp_fused_entry%/}" == "${_dmp_fused_vendor%/}" ]] && continue
        _dmp_fused_filtered_opp="${_dmp_fused_filtered_opp:+${_dmp_fused_filtered_opp}:}${_dmp_fused_entry}"
    done
    export ASCEND_CUSTOM_OPP_PATH="$_dmp_fused_vendor${_dmp_fused_filtered_opp:+:${_dmp_fused_filtered_opp}}"
fi

if [[ "$_dmp_fused_nounset_was_on" == "1" ]]; then
    set -u
fi
unset _dmp_fused_nounset_was_on _dmp_fused_script_dir _dmp_fused_vendor \
    _dmp_fused_python _dmp_fused_filtered_opp _dmp_fused_opp_entries \
    _dmp_fused_entry
