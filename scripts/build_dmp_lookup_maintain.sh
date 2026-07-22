#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DMP_LOOKUP_MAINTAIN_ROOT:-$SCRIPT_DIR/dmp-lookup-maintain}"
DEVICE="${DMP_LOOKUP_MAINTAIN_TEST_DEVICE:-npu:0}"

if [[ ! -f "$ROOT/build_and_install.sh" ]]; then
    echo "Lookup/Maintain source is missing: $ROOT" >&2
    exit 1
fi

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi

if [[ "${DMP_LOOKUP_MAINTAIN_SKIP_BUILD:-0}" != "1" ]]; then
    (
        cd "$ROOT"
        SOC_VERSION="${SOC_VERSION:-ascend910_9391}" \
        BUILD_JOBS="${BUILD_JOBS:-16}" \
        bash build_and_install.sh
    )
else
    echo "Skipping build; validating persistent Lookup/Maintain artifacts."
fi

export VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN=1
export VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT=0
export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION=0
source "$SCRIPT_DIR/dmp_lookup_maintain_runtime_env.sh"
echo "Lookup/Maintain custom OPP: $ASCEND_CUSTOM_OPP_PATH"

python3 - "$ROOT" <<'PY'
import ctypes
import os
import sys

root = sys.argv[1]
library_path = os.path.join(
    root, "opp", "vendors", "customize", "op_api", "lib", "libcust_opapi.so"
)
library = ctypes.CDLL(
    library_path,
    mode=ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0),
)
required = (
    "aclnnAsuHbmIndexLookup",
    "aclnnAsuHbmIndexLookupGetWorkspaceSize",
    "aclnnAsuHbmIndexMaintainAicpu",
    "aclnnAsuHbmIndexMaintainAicpuGetWorkspaceSize",
    "aclnnDmpLookupKvGather",
    "aclnnDmpLookupKvGatherGetWorkspaceSize",
)
missing = [name for name in required if not hasattr(library, name)]
if missing:
    raise RuntimeError(f"missing operator symbols {missing}: {library_path}")

import torch  # noqa: E402
import dmp_lookup_maintain_custom_ops  # noqa: F401,E402
import custom_ops  # noqa: F401,E402

for name in (
    "asu_hbm_index_lookup",
    "asu_hbm_index_maintain_aicpu",
    "dmp_lookup_kv_gather",
):
    if not hasattr(torch.ops.dmp_lookup_maintain, name):
        raise RuntimeError(f"operator was not registered: {name}")
for name in (
    "npu_dmp_sparse_flash_attention",
    "npu_da_attention_merge",
):
    if not hasattr(torch.ops.custom, name):
        raise RuntimeError(f"Dual-Attention operator was not registered: {name}")
print("Lookup/Maintain fixed-300 miss-only KVGather/combined-SFA registration OK")
PY

python3 "$ROOT/tests/test.py" --device "$DEVICE" --batch-size 2
echo "Standalone Lookup/Maintain operators are persistent and smoke-tested."
