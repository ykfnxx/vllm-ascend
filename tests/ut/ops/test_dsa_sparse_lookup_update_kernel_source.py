# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
KERNEL_SOURCE = (
    ROOT
    / "csrc"
    / "attention"
    / "dsa_sparse_lookup_update"
    / "op_kernel"
    / "arch35"
    / "dsa_sparse_lookup_update_simt.h"
)
KERNEL_ENTRY_SOURCE = (
    ROOT
    / "csrc"
    / "attention"
    / "dsa_sparse_lookup_update"
    / "op_kernel"
    / "dsa_sparse_lookup_update.cpp"
)
TILING_SOURCE = (
    ROOT
    / "csrc"
    / "attention"
    / "dsa_sparse_lookup_update"
    / "op_host"
    / "dsa_sparse_lookup_update_tiling.cpp"
)
BINDING_SOURCE = ROOT / "csrc" / "torch_binding.cpp"


def test_outputs_are_initialized_before_invalid_pool_row_return() -> None:
    source = KERNEL_SOURCE.read_text(encoding="utf-8")

    invalid_pool_return = source.index(
        "if (pool_entry_value < 0"
    )
    slot_initialization = source.index(
        "local_slots[local_entry] = DSA_SPARSE_NOT_FOUND"
    )
    miss_initialization = source.index(
        "local_misses[local_entry] = 0"
    )
    output_store = source.index(
        "slot_out[offset] = local_slots[local_entry]",
        invalid_pool_return,
    )

    assert slot_initialization < invalid_pool_return
    assert miss_initialization < invalid_pool_return
    assert invalid_pool_return < output_store


def test_kernel_contains_fused_maintain_and_ub_scratch() -> None:
    source = KERNEL_SOURCE.read_text(encoding="utf-8")

    assert "__ubuf__ uint32_t* protected_bits" in source
    assert "__simt_callee__ inline int32_t BlockExclusiveScan" in source
    assert "asc_atomic_or(protected_bits + word, bit)" in source
    assert "DSA_SPARSE_CLAIM_BASE" not in source
    assert "asc_atomic_cas" not in source
    assert "desired_claim" not in source
    assert "Duplicate followers" not in source
    assert "BlockExclusiveScan" in source
    assert "for (uint32_t other = 0; other < tid; ++other)" not in source
    assert "query_values[query_chunk]" in source
    assert "local_slots[query_chunk]" in source
    assert "slot_out[offset] != DSA_SPARSE_NOT_FOUND" not in source
    assert "request_free_head[0] = 0" in source
    assert "request_free_head[1]" in source
    assert "request_free_slots" in source
    assert "lru_slots" not in source


def test_only_aclnn_system_workspace_is_requested() -> None:
    kernel_source = KERNEL_ENTRY_SOURCE.read_text(encoding="utf-8")
    tiling_source = TILING_SOURCE.read_text(encoding="utf-8")

    assert "GetUserWorkspace" not in kernel_source
    assert "shared_scratch[DSA_SPARSE_UB_SCRATCH_WORDS]" in kernel_source
    assert "shared_scratch," in kernel_source
    assert "workspaceStride" not in kernel_source
    assert "platform.GetLibApiWorkSpaceSize()" in tiling_source
    assert "user_workspace_bytes" not in tiling_source
    assert "kWorkspaceStride" not in tiling_source


def test_one_vf_distributes_requests_across_physical_aivs() -> None:
    simt_source = KERNEL_SOURCE.read_text(encoding="utf-8")
    kernel_source = KERNEL_ENTRY_SOURCE.read_text(encoding="utf-8")
    tiling_source = TILING_SOURCE.read_text(encoding="utf-8")

    assert kernel_source.count("asc_vf_call<") == 1
    assert "tiling_data.reqNum," in kernel_source
    assert "gridDim.x" in simt_source
    assert "blockIdx.x" in simt_source
    assert "req_id += request_stride" in simt_source
    assert "DsaSparseLookupUpdateOneRequest(" in simt_source
    assert (
        "context->SetBlockDim(std::min("
        in tiling_source
    )
    assert "static_cast<uint32_t>(req_num), aiv_count" in tiling_source
    assert "PipeBarrier<PIPE_V>" not in kernel_source
    assert "PipeBarrier<PIPE_V>" not in simt_source


def test_zero_miss_reuse_waits_only_when_another_request_follows() -> None:
    source = KERNEL_SOURCE.read_text(encoding="utf-8")

    zero_miss = source.index("if (total_misses == 0)")
    next_request = source.index("if (reuse_scratch)", zero_miss)
    reuse_barrier = source.index("asc_syncthreads();", next_request)
    output_store = source.index(
        "slot_out[offset] = local_slots[local_entry]",
        reuse_barrier,
    )

    assert zero_miss < next_request < reuse_barrier < output_store


def test_torch_schema_matches_asu_lookup_shape() -> None:
    source = BINDING_SOURCE.read_text(encoding="utf-8")
    schema_start = source.index('"dsa_sparse_lookup_update("')
    schema_end = source.index(
        'ops.impl(\n        "dsa_sparse_lookup_update"',
        schema_start,
    )
    schema = source[schema_start:schema_end]

    for name in (
        "index",
        "slot_to_index",
        "free_slots",
        "free_head",
        "req_pool_entries",
        "query_index",
        "lookup_mask",
        "req_num",
    ):
        assert name in schema
    assert "resolved_hot_indices" not in schema
    assert "workspace" not in schema
    assert "-> (Tensor, Tensor)" in schema
