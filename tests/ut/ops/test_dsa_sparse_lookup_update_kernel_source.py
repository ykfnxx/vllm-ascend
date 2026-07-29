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
BINDING_SOURCE = ROOT / "csrc" / "torch_binding.cpp"


def test_outputs_are_initialized_before_invalid_pool_row_return() -> None:
    source = KERNEL_SOURCE.read_text(encoding="utf-8")

    invalid_pool_return = source.index(
        "if (pool_entry_value < 0"
    )
    slot_initialization = source.index(
        "slot_out[query_base + entry] = DSA_SPARSE_NOT_FOUND"
    )
    miss_initialization = source.index(
        "miss_out[query_base + entry] = 0"
    )

    assert slot_initialization < invalid_pool_return
    assert miss_initialization < invalid_pool_return


def test_kernel_contains_fused_maintain_and_hidden_workspace() -> None:
    source = KERNEL_SOURCE.read_text(encoding="utf-8")

    assert "protected_slots" in source
    assert "request_free_head[0] = 0" in source
    assert "request_free_head[1]" in source
    assert "request_free_slots" in source
    assert "lru_slots" not in source


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
