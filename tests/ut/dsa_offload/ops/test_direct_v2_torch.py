# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

import torch

from vllm_ascend.dsa_offload.ops import (
    LookupState,
    asu_kv_gather_direct_v2,
    resolve_update_batch_v2,
    turbo_resolve_update_batch_v2,
)


def test_resolve_update_v2_forwards_out_abi() -> None:
    state = LookupState(
        index=torch.empty((2, 16), dtype=torch.int32),
        slot_to_index=torch.empty((2, 8), dtype=torch.int32),
        free_slots=torch.empty((2, 2), dtype=torch.int32),
        free_head=torch.empty((2, 16), dtype=torch.int32),
    )
    rows = torch.tensor([1, 0], dtype=torch.int32)
    query_start = torch.tensor([0, 2, 3], dtype=torch.int32)
    positions = torch.tensor([128, 129, 256], dtype=torch.int32)
    topk = torch.empty((3, 1, 2048), dtype=torch.int32)
    mapped = torch.empty_like(topk)
    mask = torch.empty_like(topk)
    output = (mapped, mask)

    with patch(
        "torch.ops._C_ascend.dsa_offload_resolve_update_batch_v2",
        return_value=output,
        create=True,
    ) as operator:
        result = resolve_update_batch_v2(
            state,
            rows,
            query_start,
            positions,
            topk,
            mapped,
            mask,
            128,
            1,
        )

    operator.assert_called_once_with(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        rows,
        query_start,
        positions,
        topk,
        mapped,
        mask,
        2,
        128,
        1,
    )
    assert result is output


def test_turbo_resolve_update_v2_uses_same_tensor_abi() -> None:
    state = LookupState(
        index=torch.empty((1, 16), dtype=torch.int32),
        slot_to_index=torch.empty((1, 8), dtype=torch.int32),
        free_slots=torch.empty((1, 2), dtype=torch.int32),
        free_head=torch.empty((1, 16), dtype=torch.int32),
    )
    rows = torch.tensor([0], dtype=torch.int32)
    query_start = torch.tensor([0, 1], dtype=torch.int32)
    positions = torch.tensor([128], dtype=torch.int32)
    topk = torch.empty((1, 1, 2048), dtype=torch.int32)
    mapped = torch.empty_like(topk)
    mask = torch.empty_like(topk)

    with patch(
        "torch.ops._C_ascend.dsa_sparse_turbo_resolve_update_batch_v2",
        return_value=(mapped, mask),
        create=True,
    ) as operator:
        turbo_resolve_update_batch_v2(
            state,
            rows,
            query_start,
            positions,
            topk,
            mapped,
            mask,
            128,
            1,
        )

    assert operator.call_args.args[7] is topk
    assert operator.call_args.args[8] is mapped
    assert operator.call_args.args[9] is mask


def test_direct_gather_forwards_rank3_metadata_and_pool_table() -> None:
    destination_kv = torch.empty((4, 128, 16), dtype=torch.bfloat16)
    destination_rope = torch.empty((4, 128, 16), dtype=torch.bfloat16)
    hot_table = torch.empty((2, 2), dtype=torch.int32)
    source_kv = torch.empty((4, 128, 16), dtype=torch.bfloat16)
    source_rope = torch.empty((4, 128, 16), dtype=torch.bfloat16)
    source_table = torch.empty((2, 2), dtype=torch.int32)
    rows = torch.tensor([1, 0], dtype=torch.int32)
    query_start = torch.tensor([0, 2, 3], dtype=torch.int32)
    topk = torch.empty((3, 1, 2048), dtype=torch.int32)
    mapped = torch.empty_like(topk)
    mask = torch.empty_like(topk)

    with patch(
        "torch.ops._C_ascend.asu_kv_gather_direct_v2",
        return_value=(destination_kv, destination_rope),
        create=True,
    ) as operator:
        result = asu_kv_gather_direct_v2(
            destination_kv,
            destination_rope,
            hot_table,
            source_kv,
            source_rope,
            source_table,
            rows,
            query_start,
            topk,
            mapped,
            mask,
            128,
        )

    assert operator.call_args.args[2] is hot_table
    assert operator.call_args.args[8] is topk
    assert operator.call_args.args[9] is mapped
    assert operator.call_args.args[10] is mask
    assert result == (destination_kv, destination_rope)
