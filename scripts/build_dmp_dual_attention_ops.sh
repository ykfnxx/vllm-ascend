#!/usr/bin/env bash
set -euo pipefail

ROOT="${DMP_DUAL_ATTENTION_OP_ROOT:-/workspace/scripts/pip-cache-dual-attention}"
DEVICE="${DMP_DUAL_ATTENTION_TEST_DEVICE:-npu:0}"
RUNTIME_INSTALLER="${DMP_RUNTIME_INSTALLER:-/workspace/scripts/install_dmp_dual_attention_runtime.sh}"
RUNTIME_ENV="${DMP_RUNTIME_ENV:-/workspace/scripts/dmp_runtime_env.sh}"
OLD_OP_API="${VLLM_ASCEND_OP_API_PATH:-/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib}/libcust_opapi.so"
PACKAGE_REVISION=9
OPS=(
    gather_selection_kv_cache
    kv_select
    kv_gather
    dmp_sparse_flash_attention
    da_attention_merge
)
CMAKE_FILES=(
    config.cmake
    func.cmake
    intf.cmake
    intf_pub.cmake
    modules/Findalog.cmake
    scripts/prepare.sh
)
COMMON_HEADERS=(
    error/ops_error.h
    log/ops_log.h
    log/inner/dfx_base.h
)

if [[ ! -d "${ROOT}/op" ]]; then
    echo "Operator source not found: ${ROOT}" >&2
    exit 1
fi

for rel in "${CMAKE_FILES[@]}"; do
    if [[ ! -f "${ROOT}/op/ascendc/cmake/${rel}" ]]; then
        echo "AscendC build template missing: op/ascendc/cmake/${rel}" >&2
        echo "Reinstall the corrected DMP Dual-Attention package first." >&2
        exit 1
    fi
done

for rel in "${COMMON_HEADERS[@]}"; do
    if [[ ! -f "${ROOT}/op/ascendc/src/utils/inc/${rel}" ]]; then
        echo "AscendC common header missing: op/ascendc/src/utils/inc/${rel}" >&2
        echo "Reinstall the corrected DMP Dual-Attention package first." >&2
        exit 1
    fi
done

for op in "${OPS[@]}"; do
    if [[ ! -f "${ROOT}/op/ascendc/src/${op}/CMakeLists.txt" ]]; then
        echo "Operator source missing: op/ascendc/src/${op}" >&2
        exit 1
    fi
done

if [[ -d "${ROOT}/op/ascendc/src/sparse_flash_attention" ]]; then
    echo "Stale pre-revision-8 sparse_flash_attention source is present." >&2
    echo "Reinstall the package so the operator tree is replaced cleanly." >&2
    exit 1
fi
grep -q 'OP_ADD(DmpSparseFlashAttention)' \
    "${ROOT}/op/ascendc/src/dmp_sparse_flash_attention/op_host/dmp_sparse_flash_attention_def.cpp"
grep -q 'aclnnDmpSparseFlashAttention' \
    "${ROOT}/op/torch_ops_extension/custom_ops/csrc/npu_dmp_sparse_flash_attention.cpp"
if grep -R 'DmpDmpSparseFlashAttention' "${ROOT}/op" >/dev/null; then
    echo "Invalid duplicated DmpSparseFlashAttention prefix in operator source." >&2
    exit 1
fi

echo "DMP Dual-Attention operator package revision: ${PACKAGE_REVISION}"
echo "Building operators: ${OPS[*]}"

cd "${ROOT}"
if [[ "${DMP_DUAL_ATTENTION_SKIP_BUILD:-0}" == "1" ]]; then
    echo "Skipping OPP and wheel rebuild; validating the installed operators."
elif [[ "${DMP_DUAL_ATTENTION_OPP_ONLY:-0}" == "1" ]]; then
    echo "Rebuilding only the AscendC OPP package for ${OPP_COMPUTE_UNIT:-ascend910b}."
    OPP_OP_NAME="$(IFS=';'; echo "${OPS[*]}")" \
        OPP_COMPUTE_UNIT="${OPP_COMPUTE_UNIT:-ascend910b}" \
        bash op/scripts/build_opp.sh
elif [[ "${DMP_DUAL_ATTENTION_TORCH_ONLY:-0}" == "1" ]]; then
    echo "Rebuilding only the PyTorch extension; keeping the installed OPP package."
    CUSTOM_OPS_GATHER_ONLY= CUSTOM_OPS_SFA_ONLY= \
        bash op/scripts/build_torch_ops.sh
else
    OPP_OP_NAME="$(IFS=';'; echo "${OPS[*]}")" \
        OPP_COMPUTE_UNIT="${OPP_COMPUTE_UNIT:-ascend910b}" \
        bash op/scripts/build_opp.sh
    CUSTOM_OPS_GATHER_ONLY= CUSTOM_OPS_SFA_ONLY= \
        bash op/scripts/build_torch_ops.sh
fi

if [[ ! -f "$RUNTIME_INSTALLER" || ! -f "$RUNTIME_ENV" ]]; then
    echo "DMP runtime scripts are missing under /workspace/scripts." >&2
    exit 1
fi

DMP_DUAL_ATTENTION_OP_ROOT="$ROOT" bash "$RUNTIME_INSTALLER"
source "$RUNTIME_ENV"
NEW_OP_API="$DMP_DUAL_ATTENTION_OPP_PATH/op_api/lib/libcust_opapi.so"

cd /tmp
python3 - "$OLD_OP_API" "$NEW_OP_API" <<'PY'
import ctypes
import os
import sys

old_path, new_path = sys.argv[1:]
old = ctypes.CDLL(old_path, mode=ctypes.RTLD_GLOBAL)
new = ctypes.CDLL(new_path, mode=ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0))
assert hasattr(old, "aclnnAddRmsNormBias"), old_path
assert hasattr(old, "aclnnSparseFlashAttention"), old_path
assert hasattr(new, "aclnnKVSelect"), new_path
assert hasattr(new, "aclnnDmpSparseFlashAttention"), new_path

import custom_ops  # noqa: F401
import torch
names = (
    "npu_kv_select_out",
    "npu_kv_gather_out",
    "npu_dmp_sparse_flash_attention",
    "npu_da_attention_merge",
)
missing = [name for name in names if not hasattr(torch.ops.custom, name)]
assert not missing, f"missing custom ops: {missing}"
print("old/new operator co-load OK:", ", ".join(names))
PY

cd "${ROOT}/experiments/gather_select_kvcache"
DMP_PRELOAD_OLD_OP_API="$OLD_OP_API" python3 test_npu_kv_select_gather_correctness.py \
    --device "${DEVICE}" --batch-size 2 --max-seq-len 4096 \
    --reuse-rate 0.0 --reuse-rate 0.9 --reuse-rate 1.0
DMP_PRELOAD_OLD_OP_API="$OLD_OP_API" python3 test_npu_segmented_sfa.py \
    --device "${DEVICE}" --sweep \
    --batch-sizes 2,32 --max-seq-lens 4096 \
    --reuse-rates 0.0,0.9,1.0

echo "Dual-Attention revision ${PACKAGE_REVISION} is persistent and smoke-tested."
