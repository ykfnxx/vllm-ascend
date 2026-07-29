# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.attention.dsa_sparse import (
    DSASparseCacheConfig,
    DSASparseCohortKey,
    DSASparseLookupState,
    RequestIndexManager,
)
from vllm_ascend.dsa_sparse_constants import (
    DSA_SPARSE_FREE_HEAD_STRIDE,
    DSA_SPARSE_FREE_SLOT_COUNT,
    DSA_SPARSE_INDEX_CAPACITY,
    DSA_SPARSE_LOOKUP_SLOT_COUNT,
    DSA_SPARSE_QUERY_WIDTH,
    DSA_SPARSE_RESIDENT_SLOT_COUNT,
)


def test_cache_config_freezes_asu_dimensions_and_live_tail_layout():
    config = DSASparseCacheConfig(
        max_num_seqs=3,
        max_model_len=4096,
        block_size=128,
        index_topk=DSA_SPARSE_QUERY_WIDTH,
    )

    assert config.index_capacity == DSA_SPARSE_INDEX_CAPACITY
    assert config.resident_slot_count == DSA_SPARSE_RESIDENT_SLOT_COUNT
    assert config.free_slot_count == DSA_SPARSE_FREE_SLOT_COUNT
    assert config.lookup_slot_count == DSA_SPARSE_LOOKUP_SLOT_COUNT
    assert config.live_tail_start == DSA_SPARSE_LOOKUP_SLOT_COUNT
    assert config.hot_stride == DSA_SPARSE_LOOKUP_SLOT_COUNT + 128
    assert config.hot_blocks_per_request == 81
    assert config.total_hot_blocks == 243


def test_cache_config_rejects_non_asu_topk_and_incompatible_block_size():
    with pytest.raises(ValueError, match="index_topk"):
        DSASparseCacheConfig(
            max_num_seqs=1,
            max_model_len=4096,
            block_size=128,
            index_topk=1024,
        )
    with pytest.raises(ValueError, match="block_size"):
        DSASparseCacheConfig(
            max_num_seqs=1,
            max_model_len=4096,
            block_size=192,
            index_topk=DSA_SPARSE_QUERY_WIDTH,
        )


def test_lookup_state_uses_four_asu_tensors_and_resets_released_row():
    config = DSASparseCacheConfig(
        max_num_seqs=2,
        max_model_len=4096,
        block_size=128,
        index_topk=DSA_SPARSE_QUERY_WIDTH,
    )
    state = DSASparseLookupState.allocate(
        config,
        DSASparseCohortKey("cohort", "target"),
        device="cpu",
    )

    assert state.index.shape == (2, DSA_SPARSE_INDEX_CAPACITY)
    assert state.slot_to_index.shape == (
        2,
        DSA_SPARSE_LOOKUP_SLOT_COUNT,
    )
    assert state.free_slots.shape == (2, DSA_SPARSE_FREE_SLOT_COUNT)
    assert state.free_head.shape == (2, DSA_SPARSE_FREE_HEAD_STRIDE)
    assert state.free_slots[0, 0].item() == DSA_SPARSE_RESIDENT_SLOT_COUNT
    assert state.free_slots[0, -1].item() == DSA_SPARSE_LOOKUP_SLOT_COUNT - 1

    state.initialize_resident(0)
    assert state.index[0, 0].item() == 0
    assert state.index[0, 8191].item() == 8191
    assert state.slot_to_index[0, 0].item() == 0
    assert state.slot_to_index[0, 8191].item() == 8191

    state.index[1, 7] = 9
    state.slot_to_index[1, 3] = 7
    state.free_slots[1].fill_(-1)
    state.free_head[1].fill_(6)
    state.reset_request(1)

    assert state.index[1].eq(-1).all()
    assert state.slot_to_index[1].eq(-1).all()
    assert state.free_slots[1, 0].item() == DSA_SPARSE_RESIDENT_SLOT_COUNT
    assert state.free_slots[1, -1].item() == DSA_SPARSE_LOOKUP_SLOT_COUNT - 1
    assert state.free_head[1].eq(0).all()


def test_request_index_is_stable_and_can_be_non_contiguous():
    manager = RequestIndexManager(3)
    assert manager.acquire("a") == 0
    assert manager.acquire("b") == 1
    assert manager.acquire("c") == 2
    assert manager.release("b") == 1

    assert manager.get_index("a") == 0
    assert manager.get_index("c") == 2
