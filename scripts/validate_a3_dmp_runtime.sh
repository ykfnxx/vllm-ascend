#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_ASCEND_ENABLE_DMP_LOOKUP_MAINTAIN=1
export VLLM_ASCEND_ENABLE_DMP_FUSED_INDEXER_KV_SELECT=0
export VLLM_ASCEND_ENABLE_DMP_DUAL_ATTENTION=0
source "$SCRIPT_DIR/dmp_lookup_maintain_runtime_env.sh"

python3 - "$SCRIPT_DIR" <<'PY'
import ctypes
import importlib
import os
import sys

script_dir = sys.argv[1]
base_library = os.path.join(
    "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom",
    "vendors/vllm-ascend/op_api/lib/libcust_opapi.so",
)
dual_library = os.path.join(
    script_dir, "dmp-runtime/opp/vendors/customize/op_api/lib/libcust_opapi.so"
)
lookup_library = os.path.join(
    script_dir,
    "dmp-lookup-maintain/opp/vendors/customize/op_api/lib/libcust_opapi.so",
)

base = ctypes.CDLL(base_library, mode=ctypes.RTLD_GLOBAL)
dual = ctypes.CDLL(
    dual_library,
    mode=ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0),
)
lookup = ctypes.CDLL(
    lookup_library,
    mode=ctypes.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0),
)

required_symbols = {
    base_library: ("aclnnAddRmsNormBias", "aclnnSparseFlashAttention"),
    dual_library: ("aclnnDmpSparseFlashAttention", "aclnnDaAttentionMerge"),
    lookup_library: (
        "aclnnAsuHbmIndexLookup",
        "aclnnAsuHbmIndexMaintainAicpu",
        "aclnnDmpLookupKvGather",
    ),
}
for library_path, symbols in required_symbols.items():
    handle = {base_library: base, dual_library: dual, lookup_library: lookup}[
        library_path
    ]
    missing = [name for name in symbols if not hasattr(handle, name)]
    if missing:
        raise RuntimeError(f"missing symbols {missing}: {library_path}")

from transformers import AutoConfig

AutoConfig.for_model("glm_moe_dsa")

import torch
import torch_npu

base_extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
required_base_ops = (
    "moe_gating_top_k",
    "npu_moe_init_routing_custom",
    "npu_lightning_indexer_quant",
    "npu_sparse_flash_attention",
    "npu_add_rms_norm_bias",
)
missing_base_ops = [
    name for name in required_base_ops if not hasattr(torch.ops._C_ascend, name)
]
if missing_base_ops:
    raise RuntimeError(
        f"A3 base extension is missing PyTorch operators: {missing_base_ops}"
    )

import custom_ops  # noqa: F401
import dmp_lookup_maintain_custom_ops  # noqa: F401

required_torch_ops = (
    (torch.ops.custom, "npu_dmp_sparse_flash_attention"),
    (torch.ops.custom, "npu_da_attention_merge"),
    (torch.ops.dmp_lookup_maintain, "asu_hbm_index_lookup"),
    (torch.ops.dmp_lookup_maintain, "asu_hbm_index_maintain_aicpu"),
    (torch.ops.dmp_lookup_maintain, "dmp_lookup_kv_gather"),
)
missing_ops = [name for namespace, name in required_torch_ops if not hasattr(namespace, name)]
if missing_ops:
    raise RuntimeError(f"missing registered torch operators: {missing_ops}")

print("A3_DMP_RUNTIME_VALIDATION_OK")
print("device:", torch_npu.npu.get_device_name(0))
print("SoC:", torch_npu.npu.get_soc_version())
print("base extension:", base_extension.__file__)
print("base operators:", ", ".join(required_base_ops))
print("registered operators:", ", ".join(name for _, name in required_torch_ops))
PY
