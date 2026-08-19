# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = ROOT / "csrc" / "attention" / "dsa_sparse_lookup_update_batch"
KERNEL_SOURCE = OP_ROOT / "op_kernel" / "arch35" / "dsa_sparse_lookup_update_batch_simt.h"
KERNEL_ENTRY_SOURCE = OP_ROOT / "op_kernel" / "dsa_sparse_lookup_update_batch.cpp"
TILING_SOURCE = OP_ROOT / "op_host" / "dsa_sparse_lookup_update_batch_tiling.cpp"
BINDING_SOURCE = ROOT / "csrc" / "torch_binding.cpp"
OLD_OP_ROOT = ROOT / "csrc" / "attention" / "dsa_sparse_lookup_update"


def test_batch_operator_is_isolated_from_the_single_query_operator():
    assert OP_ROOT.is_dir()
    assert OLD_OP_ROOT.is_dir()
    source = BINDING_SOURCE.read_text(encoding="utf-8")

    assert '"dsa_sparse_lookup_update("' in source
    assert '"dsa_sparse_lookup_update_batch("' in source
    assert "Tensor query_start_loc" in source


def test_kernel_keeps_one_protected_bitmap_across_packed_queries():
    source = KERNEL_SOURCE.read_text(encoding="utf-8")
    clear_position = source.index("protected_bits[word] = 0U")
    query_loop = source.index("for (uint32_t query_id =", clear_position)
    process_query = source.index("ProcessQuery(", query_loop)

    assert clear_position < query_loop < process_query
    assert "query_start_loc[req_id]" in source
    assert "query_start_loc[req_id + 1U]" in source
    assert "ProtectSlot(protected_bits" in source
    assert "req_id += request_stride" in source


def test_overflow_defaults_are_written_before_request_validation():
    source = KERNEL_SOURCE.read_text(encoding="utf-8")
    initializer = source.index("InitializeQueryRange(")
    request_validation = source.index("if (pool_entry_value < 0")

    assert initializer < request_validation
    assert "DSA_SPARSE_BATCH_FALLBACK_SLOT" in source
    assert "miss_out[offset] = 0" in source


def test_batch_operator_uses_ub_scratch_and_system_workspace_only():
    kernel_source = KERNEL_ENTRY_SOURCE.read_text(encoding="utf-8")
    tiling_source = TILING_SOURCE.read_text(encoding="utf-8")

    assert "shared_scratch[DSA_SPARSE_BATCH_UB_SCRATCH_WORDS]" in kernel_source
    assert "GetUserWorkspace" not in kernel_source
    assert "platform.GetLibApiWorkSpaceSize()" in tiling_source
    assert "user_workspace_bytes" not in tiling_source
