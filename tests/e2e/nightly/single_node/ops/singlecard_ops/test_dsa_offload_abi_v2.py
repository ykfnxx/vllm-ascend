# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import copy

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.dsa_offload.ops.reference import (
    INVALID_INDEX,
    ReferenceState,
    resolve_update_v2_reference,
)
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

INDEX_CAPACITY = 128 * 1024
RESIDENT_SLOTS = 8 * 1024
LOOKUP_SLOTS = 10 * 1024
QUERY_WIDTH = 2 * 1024
FREE_HEAD_STRIDE = 16
BLOCK_SIZE = 128


def _make_state() -> ReferenceState:
    index = [[INVALID_INDEX] * INDEX_CAPACITY]
    slot_to_index = [[INVALID_INDEX] * LOOKUP_SLOTS]
    for token in range(RESIDENT_SLOTS):
        index[0][token] = token
        slot_to_index[0][token] = token
    return ReferenceState(
        index=index,
        slot_to_index=slot_to_index,
        free_slots=[list(range(RESIDENT_SLOTS, LOOKUP_SLOTS))],
        free_head=[[0] * FREE_HEAD_STRIDE],
    )


def _to_npu_state(state: ReferenceState) -> list[torch.Tensor]:
    return [
        torch.tensor(state.index, dtype=torch.int32, device="npu"),
        torch.tensor(state.slot_to_index, dtype=torch.int32, device="npu"),
        torch.tensor(state.free_slots, dtype=torch.int32, device="npu"),
        torch.tensor(state.free_head, dtype=torch.int32, device="npu"),
    ]


def _make_case(decode_mode: int) -> tuple[list[int], list[list[int]]]:
    if decode_mode == 0:
        return [16512], [[1, 9000, 16512, 16513, -1]]
    return (
        [16512, 16513],
        [
            [1, 9000, 16511, 16512, 16513],
            [9000, 16511, 16512, 16513, 16514],
        ],
    )


@pytest.mark.parametrize(
    "operator_name,decode_mode",
    [
        ("dsa_offload_resolve_update_batch_v2", 0),
        ("dsa_offload_resolve_update_batch_v2", 1),
        ("dsa_sparse_turbo_resolve_update_batch_v2", 1),
    ],
)
@torch.inference_mode()
def test_resolve_update_v2_matches_reference(
    operator_name: str,
    decode_mode: int,
) -> None:
    positions, active_topk = _make_case(decode_mode)
    query_num = len(positions)
    semantic_topk = [
        row + [INVALID_INDEX] * (QUERY_WIDTH - len(row))
        for row in active_topk
    ]
    expected_state = _make_state()
    expected_mapped, expected_gather = resolve_update_v2_reference(
        expected_state,
        [0],
        [0, query_num],
        positions,
        semantic_topk,
        block_size=BLOCK_SIZE,
        decode_mode=decode_mode,
    )

    initial_state = _make_state()
    index, slot_to_index, free_slots, free_head = _to_npu_state(
        copy.deepcopy(initial_state)
    )
    request_rows = torch.tensor([0], dtype=torch.int32, device="npu")
    query_start_loc = torch.tensor(
        [0, query_num], dtype=torch.int32, device="npu"
    )
    query_positions = torch.tensor(
        positions, dtype=torch.int32, device="npu"
    )
    topk = torch.tensor(
        semantic_topk, dtype=torch.int32, device="npu"
    ).view(query_num, 1, QUERY_WIDTH)
    mapped_out = torch.empty_like(topk)
    gather_out = torch.empty_like(topk)

    mapped, gather = getattr(torch.ops._C_ascend, operator_name)(
        index,
        slot_to_index,
        free_slots,
        free_head,
        request_rows,
        query_start_loc,
        query_positions,
        topk,
        mapped_out,
        gather_out,
        1,
        BLOCK_SIZE,
        decode_mode,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        mapped.cpu().view(query_num, QUERY_WIDTH),
        torch.tensor(expected_mapped, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        gather.cpu().view(query_num, QUERY_WIDTH),
        torch.tensor(expected_gather, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    for actual, expected in zip(
        (index, slot_to_index, free_slots, free_head),
        (
            expected_state.index,
            expected_state.slot_to_index,
            expected_state.free_slots,
            expected_state.free_head,
        ),
    ):
        torch.testing.assert_close(
            actual.cpu(),
            torch.tensor(expected, dtype=torch.int32),
            rtol=0,
            atol=0,
        )


@torch.inference_mode()
def test_direct_gather_uses_pool_row_and_returns_cache_aliases() -> None:
    destination_kv = torch.zeros(
        (4, BLOCK_SIZE, 16), dtype=torch.bfloat16, device="npu"
    )
    destination_rope = torch.zeros_like(destination_kv)
    source_kv = torch.zeros_like(destination_kv)
    source_rope = torch.zeros_like(destination_kv)
    expected_kv = torch.arange(16, dtype=torch.float32).to(torch.bfloat16)
    expected_rope = expected_kv + 32
    source_kv[3, 5].copy_(expected_kv.to("npu"))
    source_rope[3, 5].copy_(expected_rope.to("npu"))

    hot_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="npu")
    source_table = torch.tensor(
        [[0, 0], [3, 0]], dtype=torch.int32, device="npu"
    )
    request_rows = torch.tensor([1, -1], dtype=torch.int32, device="npu")
    query_start_loc = torch.tensor([0, 1, 1], dtype=torch.int32, device="npu")
    semantic_topk = torch.full(
        (1, 1, QUERY_WIDTH), INVALID_INDEX, dtype=torch.int32, device="npu"
    )
    mapped_indices = torch.full_like(semantic_topk, INVALID_INDEX)
    gather_mask = torch.zeros_like(semantic_topk)
    semantic_topk[0, 0, 0] = 5
    mapped_indices[0, 0, 0] = 7
    gather_mask[0, 0, 0] = 1

    result_kv, result_rope = torch.ops._C_ascend.asu_kv_gather_direct_v2(
        destination_kv,
        destination_rope,
        hot_table,
        source_kv,
        source_rope,
        source_table,
        request_rows,
        query_start_loc,
        semantic_topk,
        mapped_indices,
        gather_mask,
        BLOCK_SIZE,
        2,
    )
    torch.npu.synchronize()

    assert result_kv.data_ptr() == destination_kv.data_ptr()
    assert result_rope.data_ptr() == destination_rope.data_ptr()
    torch.testing.assert_close(
        result_kv[2, 7].cpu(), expected_kv, rtol=0, atol=0
    )
    torch.testing.assert_close(
        result_rope[2, 7].cpu(), expected_rope, rtol=0, atol=0
    )
