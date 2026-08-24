# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch

from vllm_ascend.attention.dsa_sparse_shm import (
    DSASparseSharedMemoryStore,
)


def test_shared_memory_round_trip_and_unlink(tmp_path):
    store = DSASparseSharedMemoryStore(tmp_path)
    indexer = torch.arange(
        4 * 2 * 3,
        dtype=torch.bfloat16,
    ).reshape(4, 2, 1, 3)
    main = torch.arange(
        4 * 2 * 2,
        dtype=torch.bfloat16,
    ).reshape(4, 2, 1, 2)

    payload = store.publish(
        cache_kind="indexer",
        cache_layer_name="model.layers.0.self_attn.indexer.k_cache",
        cache=(indexer,),
        cache_block_ids=torch.tensor([2, 0], dtype=torch.int64),
        logical_num_blocks=4,
        main_cache=(main,),
        main_tail_block_id=2,
        tail_valid_count=1,
    )

    assert payload is not None
    assert (tmp_path / payload.name).exists()
    assert payload.cache_kind == "indexer"
    assert len(payload.cache_planes) == 1
    assert len(payload.tail_planes) == 1

    with store.open(payload) as reader:
        indexer_source = reader.tensor(payload.cache_planes[0])
        tail_source = reader.tensor(payload.tail_planes[0])
        assert torch.equal(indexer_source, indexer.index_select(0, torch.tensor([2, 0])))
        assert torch.equal(tail_source, main[2, :1])
        reader.unlink()

    assert not (tmp_path / payload.name).exists()
    # Reader tensors own CPU copies and remain valid after the mmap closes.
    assert torch.equal(indexer_source, indexer.index_select(0, torch.tensor([2, 0])))
    assert torch.equal(tail_source, main[2, :1])


def test_shared_memory_preserves_two_plane_c8_layout(tmp_path):
    float8_dtype = torch.float8_e4m3fn
    store = DSASparseSharedMemoryStore(tmp_path)
    indexer_k = torch.zeros((2, 2, 1, 4), dtype=float8_dtype)
    indexer_scale = torch.ones((2, 2, 1, 1), dtype=torch.float32)
    main = torch.zeros((2, 2, 1, 3), dtype=torch.bfloat16)

    payload = store.publish(
        cache_kind="indexer",
        cache_layer_name="model.layers.0.self_attn.indexer.k_cache",
        cache=(indexer_k, indexer_scale),
        cache_block_ids=torch.tensor([1], dtype=torch.int64),
        logical_num_blocks=2,
        main_cache=(main,),
        main_tail_block_id=None,
        tail_valid_count=0,
    )

    assert payload is not None
    assert [plane.dtype for plane in payload.cache_planes] == [
        "float8_e4m3fn",
        "float32",
    ]
    with store.open(payload) as reader:
        packed = reader.tensor(payload.cache_planes[0])
        scale = reader.tensor(payload.cache_planes[1])
        assert packed.dtype == float8_dtype
        assert scale.dtype == torch.float32
        del packed, scale
        reader.unlink()


def test_shared_memory_round_trip_for_mtp_draft(tmp_path):
    store = DSASparseSharedMemoryStore(tmp_path)
    draft = torch.arange(
        4 * 2 * 3,
        dtype=torch.bfloat16,
    ).reshape(4, 2, 1, 3)

    payload = store.publish(
        cache_kind="mtp_draft",
        cache_layer_name="model.layers.1.self_attn.attn",
        cache=(draft,),
        cache_block_ids=torch.tensor([3, 1], dtype=torch.int64),
        logical_num_blocks=4,
    )

    assert payload is not None
    assert payload.cache_kind == "mtp_draft"
    assert not payload.tail_planes
    with store.open(payload) as reader:
        source = reader.tensor(payload.cache_planes[0])
        assert torch.equal(
            source,
            draft.index_select(0, torch.tensor([3, 1])),
        )
        reader.unlink()
