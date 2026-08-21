# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.dsa_sparse import DSASparseCoordinator
from vllm_ascend.dsa_sparse_backend import (
    DSASparseStorageKeyEncoder,
    MockDSASparseKVBackend,
)
from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_FREE_HEAD_STRIDE,
    DSA_SPARSE_FREE_SLOT_COUNT,
    DSA_SPARSE_INDEX_CAPACITY,
    DSA_SPARSE_LOOKUP_SLOT_COUNT,
    DSA_SPARSE_QUERY_WIDTH,
)


def make_coordinator(
    leader: DSASparseCoordinator | None = None,
    *,
    mtp_enabled: bool = False,
    max_verify_tokens_per_request: int = 1,
) -> DSASparseCoordinator:
    return DSASparseCoordinator(
        max_num_seqs=3,
        block_size=128,
        plane_layouts=((torch.bfloat16, (1, 576)),),
        device="cpu",
        leader=leader,
        mtp_enabled=mtp_enabled,
        max_verify_tokens_per_request=max_verify_tokens_per_request,
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


def test_mtp_coordinator_rejects_an_empty_verify_region():
    with pytest.raises(ValueError, match="verify capacity"):
        make_coordinator(
            mtp_enabled=True,
            max_verify_tokens_per_request=0,
        )


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


def test_normal_decode_puts_only_when_tail_becomes_full():
    backend = MockDSASparseKVBackend()
    coordinator = DSASparseCoordinator(
        max_num_seqs=1,
        block_size=128,
        plane_layouts=(
            (torch.float32, (1,)),
            (torch.float32, (1,)),
        ),
        device="cpu",
        backend=backend,
        storage_key_encoder=DSASparseStorageKeyEncoder(),
    )
    backend.register_layer_cache(
        layer_id=0,
        block_size=128,
        cache_planes=coordinator.hot_main_cache,
    )
    coordinator.set_request_block_hashes(0, [b"block-0"])

    coordinator.commit_decode_tail("model.layers.0.self_attn.attn", [0], [126])
    assert not backend.put_calls

    coordinator.commit_decode_tail("model.layers.0.self_attn.attn", [0], [127])
    assert len(backend.put_calls) == 1
    assert backend.put_calls[0][2].tolist() == [80]


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


def test_mtp_main_write_uses_per_request_verify_staging():
    coordinator = make_coordinator(
        mtp_enabled=True,
        max_verify_tokens_per_request=4,
    )
    slots = coordinator.build_main_slot_mapping_batch(
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 2, 5], dtype=torch.int32),
        torch.tensor([100, 101, 200, 201, 202], dtype=torch.int64),
    )

    assert slots.tolist() == [
        10369,
        10370,
        2 * 10496 + 10369,
        2 * 10496 + 10370,
        2 * 10496 + 10371,
    ]
    assert slots.dtype == torch.int32


def test_mtp_main_write_waits_for_the_previous_store():
    coordinator = make_coordinator(
        mtp_enabled=True,
        max_verify_tokens_per_request=4,
    )

    with patch.object(coordinator, "wait_for_store") as wait_for_store:
        coordinator.build_main_slot_mapping_batch(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([100], dtype=torch.int64),
        )

    wait_for_store.assert_called_once_with()


def test_mtp_leader_resolves_packed_history_and_staging_once():
    leader = make_coordinator(
        mtp_enabled=True,
        max_verify_tokens_per_request=4,
    )
    follower = make_coordinator(
        leader,
        mtp_enabled=True,
        max_verify_tokens_per_request=4,
    )
    req_pool_entries = torch.tensor([0, 2], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    query_positions = torch.tensor([130, 131, 260], dtype=torch.int64)
    topk = torch.full(
        (3, 1, DSA_SPARSE_QUERY_WIDTH),
        -1,
        dtype=torch.int64,
    )
    topk[0, 0, :4] = torch.tensor([10, 128, 130, -1])
    topk[1, 0, :4] = torch.tensor([10, 128, 130, 132])
    topk[2, 0, :3] = torch.tensor([20, 256, 260])
    calls = []

    def fake_lookup(*args):
        calls.append(args)
        query_index = args[6]
        lookup_mask = args[7]
        return torch.full_like(query_index, 77), lookup_mask.clone()

    with patch(
        "vllm_ascend.attention.dsa_sparse.dsa_sparse_lookup_update_batch",
        side_effect=fake_lookup,
    ):
        leader.resolve_batch(
            topk,
            req_pool_entries,
            query_start_loc,
            query_positions,
        )
        follower.reuse_leader_plan(req_pool_entries)

    assert len(calls) == 1
    assert calls[0][4] is req_pool_entries
    assert calls[0][5] is query_start_loc
    assert calls[0][7][0, :4].tolist() == [1, 0, 0, 0]
    assert calls[0][7][1, :4].tolist() == [1, 0, 0, 0]
    assert calls[0][7][2, :3].tolist() == [1, 0, 0]
    assert leader.attention_indices[0, :4].tolist() == [
        77,
        10240,
        10369,
        -1,
    ]
    assert leader.attention_indices[1, :4].tolist() == [
        77,
        10240,
        10369,
        -1,
    ]
    assert leader.attention_indices[2, :3].tolist() == [
        77,
        10240,
        10369,
    ]
    assert follower.attention_indices is leader.attention_indices
    assert follower.query_start_loc is leader.query_start_loc
    assert follower.query_positions is leader.query_positions


def test_mtp_request_reset_zeros_tail_and_fallback_only():
    coordinator = DSASparseCoordinator(
        max_num_seqs=2,
        block_size=128,
        plane_layouts=((torch.bfloat16, (1,)),),
        device="cpu",
        mtp_enabled=True,
        max_verify_tokens_per_request=4,
    )
    flat_cache = coordinator.hot_main_cache[0].view(-1)
    row_base = coordinator.request_row_stride
    flat_cache[row_base + coordinator.fallback_zero_slot] = 9
    flat_cache[row_base + coordinator.verify_staging_base] = 7
    flat_cache[row_base + coordinator.tail_base : row_base + coordinator.tail_base + coordinator.block_size] = 5

    coordinator.reset_hot_request(1)

    assert flat_cache[row_base + coordinator.fallback_zero_slot].item() == 0
    assert (
        flat_cache[row_base + coordinator.tail_base : row_base + coordinator.tail_base + coordinator.block_size]
        .eq(0)
        .all()
    )
    assert flat_cache[row_base + coordinator.verify_staging_base].item() == 7


def test_mtp_mock_store_reports_only_the_accepted_prefix_lengths():
    coordinator = make_coordinator(
        mtp_enabled=True,
        max_verify_tokens_per_request=4,
    )
    coordinator.query_start_loc = torch.tensor(
        [0, 2, 5],
        dtype=torch.int32,
    )
    coordinator.query_positions = torch.tensor(
        [100, 101, 200, 201, 202],
        dtype=torch.int64,
    )
    coordinator.req_pool_entries = torch.tensor(
        [0, 2],
        dtype=torch.int32,
    )
    accepted = torch.tensor([1, 3], dtype=torch.int32)

    with (
        patch(
            "vllm_ascend.attention.dsa_sparse.dsa_sparse_probe.is_enabled",
            return_value=True,
        ),
        patch(
            "vllm_ascend.attention.dsa_sparse.dsa_sparse_probe.synchronize_device",
        ),
        patch(
            "vllm_ascend.attention.dsa_sparse.dsa_sparse_probe.emit",
        ) as emit,
    ):
        coordinator.commit_accepted_to_tail(
            "model.layers.0.self_attn.attn",
            [0, 2, 5],
            [100, 101, 200, 201, 202],
            accepted.tolist(),
            [0, 2],
        )

    assert emit.call_args.kwargs["accepted_input_kv_count"] == [1, 3]
    assert emit.call_args.kwargs["committed_kv_count"] == 4


def test_mtp_commit_puts_full_tail_without_copying_rejected_suffix():
    backend = MockDSASparseKVBackend()
    coordinator = DSASparseCoordinator(
        max_num_seqs=1,
        block_size=128,
        plane_layouts=(
            (torch.float32, (1,)),
            (torch.float32, (1,)),
        ),
        device="cpu",
        mtp_enabled=True,
        max_verify_tokens_per_request=3,
        backend=backend,
        storage_key_encoder=DSASparseStorageKeyEncoder(),
    )
    backend.register_layer_cache(
        layer_id=0,
        block_size=128,
        cache_planes=coordinator.hot_main_cache,
    )
    coordinator.set_request_block_hashes(0, [b"block-0"])
    coordinator.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    coordinator.query_positions = torch.tensor([126, 127, 128], dtype=torch.int64)
    coordinator.req_pool_entries = torch.tensor([0], dtype=torch.int32)

    for plane in coordinator.hot_main_cache:
        flat = plane.view(-1)
        flat[coordinator.tail_base : coordinator.tail_base + 126].fill_(1)
        flat[coordinator.verify_staging_base : coordinator.verify_staging_base + 3] = torch.tensor(
            [2, 3, 9], dtype=flat.dtype
        )

    coordinator.commit_accepted_to_tail(
        "model.layers.0.self_attn.attn",
        [0, 3],
        [126, 127, 128],
        [2],
        [0],
    )

    assert len(backend.put_calls) == 1
    assert backend.put_calls[0][0] == 0
    assert backend.put_calls[0][2].tolist() == [80]
    for plane in coordinator.hot_main_cache:
        flat = plane.view(-1)
        assert flat[coordinator.tail_base + 126].item() == 2
        assert flat[coordinator.tail_base + 127].item() == 3
        assert flat[coordinator.tail_base].item() == 1
