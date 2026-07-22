#!/usr/bin/env bash

# Source all runtime settings needed by the DMP Dual-Attention entry scripts.
_dmp_nounset_was_on=0
case "$-" in
    *u*) _dmp_nounset_was_on=1 ;;
esac
set +u
if [[ "${VLLM_ASCEND_DMP_KV_BACKEND:-local}" == "hixl" ]]; then
    _dmp_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$_dmp_script_dir/dmp_hixl_runtime_env.sh"
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export VISIBLE_DEVICES="${VISIBLE_DEVICES:-5}"
export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export VLLM_ASCEND_ENABLE_DMP=1
export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION=1
# DMP Dual-Attention stream topology:
# two: S0=A/B indexer + hit/miss SFA + merge + MLP; S1=KVSelect + KVGather.
# four: S0=main compute, S1=A indexer, plus separate Select/Gather streams.
export VLLM_ASCEND_DMP_STREAM_MODE="${VLLM_ASCEND_DMP_STREAM_MODE:-two}"
export VLLM_ASCEND_DMP_KV_BACKEND="${VLLM_ASCEND_DMP_KV_BACKEND:-local}"
_dmp_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_ASCEND_DMP_HIXL_CONFIG="${VLLM_ASCEND_DMP_HIXL_CONFIG:-$_dmp_script_dir/dmp_hixl_config.json}"
export DMP_DUAL_ATTENTION_RUNTIME_ROOT="${DMP_DUAL_ATTENTION_RUNTIME_ROOT:-$_dmp_script_dir/dmp-runtime}"
_dmp_persistent_vendor="$DMP_DUAL_ATTENTION_RUNTIME_ROOT/opp/vendors/customize"
export DMP_DUAL_ATTENTION_OPP_PATH="${DMP_DUAL_ATTENTION_OPP_PATH:-$_dmp_persistent_vendor}"
DMP_DUAL_ATTENTION_OPP_PATH="${DMP_DUAL_ATTENTION_OPP_PATH%/}"
export DMP_DUAL_ATTENTION_OPP_PATH
export PYTHONPATH="$DMP_DUAL_ATTENTION_RUNTIME_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$VLLM_ASCEND_DMP_KV_BACKEND" == "hixl" ]]; then
    export DMP_HIXL_PYTHON_PATH="${DMP_HIXL_PYTHON_PATH:-$DMP_DUAL_ATTENTION_RUNTIME_ROOT/hixl-python}"
    export PYTHONPATH="$DMP_HIXL_PYTHON_PATH:$PYTHONPATH"
fi

VLLM_ASCEND_OP_API_PATH="${VLLM_ASCEND_OP_API_PATH:-}"
if [[ -z "$VLLM_ASCEND_OP_API_PATH" ]]; then
    for vendor_root in \
        /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend \
        /usr/local/python3.11.14/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend; do
        if [[ -f "$vendor_root/op_api/lib/libcust_opapi.so" ]]; then
            VLLM_ASCEND_OP_API_PATH="$vendor_root/op_api/lib"
            break
        fi
    done
fi

if [[ ! -f "$VLLM_ASCEND_OP_API_PATH/libcust_opapi.so" ]]; then
    echo "vllm-ascend libcust_opapi.so not found; checked bundled package paths." >&2
    return 1 2>/dev/null || exit 1
fi
export VLLM_ASCEND_OP_API_PATH

# The old extension opens libcust_opapi.so by name, so its directory must be
# first. The dual-attention extension loads its own library by absolute path.
filtered_ld_path=""
IFS=':' read -r -a ld_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_entries[@]}"; do
    [[ -n "$entry" ]] || continue
    [[ "${entry%/}" == "${VLLM_ASCEND_OP_API_PATH%/}" ]] && continue
    [[ "${entry%/}" == "$DMP_DUAL_ATTENTION_OPP_PATH/op_api/lib" ]] && continue
    [[ "${entry%/}" == "/usr/local/Ascend/cann-8.5.1/opp/vendors/customize/op_api/lib" ]] && continue
    filtered_ld_path="${filtered_ld_path:+${filtered_ld_path}:}${entry}"
done
export LD_LIBRARY_PATH="$VLLM_ASCEND_OP_API_PATH${filtered_ld_path:+:${filtered_ld_path}}"

filtered_opp_path=""
IFS=':' read -r -a opp_entries <<< "${ASCEND_CUSTOM_OPP_PATH:-}"
for entry in "${opp_entries[@]}"; do
    [[ -n "$entry" ]] || continue
    [[ "${entry%/}" == "${DMP_DUAL_ATTENTION_OPP_PATH%/}" ]] && continue
    [[ "${entry%/}" == "/usr/local/Ascend/cann-8.5.1/opp/vendors/customize" ]] && continue
    filtered_opp_path="${filtered_opp_path:+${filtered_opp_path}:}${entry}"
done
export ASCEND_CUSTOM_OPP_PATH="$DMP_DUAL_ATTENTION_OPP_PATH${filtered_opp_path:+:${filtered_opp_path}}"

if [[ "$_dmp_nounset_was_on" == "1" ]]; then
    set -u
fi
unset _dmp_nounset_was_on _dmp_script_dir _dmp_persistent_vendor filtered_ld_path filtered_opp_path ld_entries opp_entries entry vendor_root
