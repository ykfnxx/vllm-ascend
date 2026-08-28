# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

import torch

from vllm_ascend.attention.dsa_sparse import DSASparseCoordinator
from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_FREE_HEAD_STRIDE,
    DSA_SPARSE_FREE_SLOT_COUNT,
    DSA_SPARSE_INDEX_CAPACITY,
    DSA_SPARSE_LOOKUP_SLOT_COUNT,
    DSA_SPARSE_QUERY_WIDTH,
)


def make_coordinator(
    leader: DSASparseCoordinator | None = None,
) -> DSASparseCoordinator:
    return DSASparseCoordinator(
        max_num_seqs=3,
        block_size=128,
        plane_layouts=((torch.bfloat16, (1, 576)),),
        device="cpu",
        leader=leader,
    )


def semantic_topk() -> torch.Tensor:
    topk = torch.full(
        (2, 1, DSA_SPARSE_QUERY_WIDTH),
        -1,
        dtype=torch.int64,
    )
    topk[0, 0, :2] = torch.tensor([10, 130])
    topk[1, 0, :2] = torch.tensor([20, 258])
    return topk


def test_coordinator_owns_per_layer_hot_cache_and_leader_lookup_state():
    leader = make_coordinator()
    follower = make_coordinator(leader)

    assert leader.hot_main_cache[0].shape == (243, 128, 1, 576)
    assert follower.hot_main_cache[0].shape == (243, 128, 1, 576)
    assert follower.hot_main_cache[0] is not leader.hot_main_cache[0]
    assert leader.index.shape == (3, DSA_SPARSE_INDEX_CAPACITY)
    assert leader.slot_to_index.shape == (3, DSA_SPARSE_LOOKUP_SLOT_COUNT)
    assert leader.free_slots.shape == (3, DSA_SPARSE_FREE_SLOT_COUNT)
    assert leader.free_head.shape == (3, DSA_SPARSE_FREE_HEAD_STRIDE)
    assert follower.index is None
    assert follower.slot_to_index is None
    assert follower.free_slots is None
    assert follower.free_head is None


def test_request_initialization_and_release_reset_the_leader_row():
    coordinator = make_coordinator()

    coordinator.initialize_request(1)
    assert coordinator.index[1, 0].item() == 0
    assert coordinator.index[1, 8191].item() == 8191
    assert coordinator.slot_to_index[1, 8191].item() == 8191
    assert coordinator.free_slots[1, 0].item() == 8192
    assert coordinator.free_slots[1, -1].item() == 10239

    coordinator.index[1, 9000] = 7
    coordinator.free_head[1].fill_(9)
    coordinator.reset_request(1)
    assert coordinator.index[1].eq(-1).all()
    assert coordinator.slot_to_index[1].eq(-1).all()
    assert coordinator.free_slots[1, 0].item() == 8192
    assert coordinator.free_slots[1, -1].item() == 10239
    assert coordinator.free_head[1].eq(0).all()


def test_request_initialization_maps_custom_residents_to_stable_slots():
    coordinator = make_coordinator()

    coordinator.initialize_request(1, [255, 7, 3, 128])

    assert coordinator.index[1, 255].item() == 0
    assert coordinator.index[1, 7].item() == 1
    assert coordinator.index[1, 3].item() == 2
    assert coordinator.index[1, 128].item() == 3
    assert coordinator.index[1, 0].item() == -1
    assert coordinator.slot_to_index[1, :5].tolist() == [
        255,
        7,
        3,
        128,
        -1,
    ]


def test_main_write_uses_stable_pool_entry_and_live_tail_offset():
    coordinator = make_coordinator()
    slots = coordinator.build_main_slot_mapping(
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([131, 261], dtype=torch.int32),
    )

    assert slots.tolist() == [10242, 2 * 10368 + 10244]
    assert slots.dtype == torch.int32


def test_leader_resolves_once_and_follower_reuses_the_same_plan():
    leader = make_coordinator()
    follower = make_coordinator(leader)
    req_pool_entries = torch.tensor([0, 2], dtype=torch.int32)
    seq_lens = torch.tensor([131, 261], dtype=torch.int32)
    calls = []

    def fake_lookup(*args):
        calls.append(args)
        query_index = args[5]
        lookup_mask = args[6]
        return torch.full_like(query_index, 77), lookup_mask.clone()

    with patch(
        "vllm_ascend.attention.dsa_sparse.dsa_sparse_lookup_update",
        side_effect=fake_lookup,
    ):
        leader.resolve(semantic_topk(), req_pool_entries, seq_lens)
        follower.reuse_leader_plan(req_pool_entries)

    assert len(calls) == 1
    assert calls[0][4] is req_pool_entries
    assert calls[0][5].shape == (2, DSA_SPARSE_QUERY_WIDTH)
    assert calls[0][6][0, :3].tolist() == [1, 0, 0]
    assert calls[0][6][1, :3].tolist() == [1, 0, 0]
    assert leader.attention_indices[0, :3].tolist() == [77, 10242, -1]
    assert leader.attention_indices[1, :3].tolist() == [77, 10242, -1]
    assert leader.hot_block_table.shape == (2, 81)
    assert leader.hot_block_table[0, :2].tolist() == [0, 1]
    assert leader.hot_block_table[1, :2].tolist() == [162, 163]
    assert follower.attention_indices is leader.attention_indices
    assert follower.hot_block_table is leader.hot_block_table
    assert follower.slot_out is leader.slot_out
    assert follower.miss_out is leader.miss_out
