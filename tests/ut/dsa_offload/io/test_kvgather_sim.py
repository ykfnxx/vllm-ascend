# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

import torch

from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout
from vllm_ascend.dsa_offload.kvgather_sim import KVGatherSimBackend


def test_dense_gather_uses_registered_cache_and_synthetic_source() -> None:
    layout = HotCacheLayout(128, 2, 2)
    backend = KVGatherSimBackend(layout)
    kv = torch.ones((layout.hot_blocks, 128, 1, 16), dtype=torch.bfloat16)
    rope = torch.ones((layout.hot_blocks, 128, 1, 16), dtype=torch.bfloat16)
    backend.register_get_cache(
        layer_id=3,
        block_size=128,
        cache_planes=(kv, rope),
    )
    block_table = layout.block_table(torch.tensor([0, 1], dtype=torch.int32))
    request_rows = torch.tensor([0, 1], dtype=torch.int32)
    positions = torch.tensor([[10, 130], [20, 260]], dtype=torch.int32)
    slots = torch.tensor([[7, 129], [9, 131]], dtype=torch.int32)
    miss_mask = torch.ones_like(positions)

    with patch(
        "vllm_ascend.dsa_offload.kvgather_sim.asu_kv_gather"
    ) as gather:
        handled = backend.gather_history_misses(
            layer_id=3,
            destination_block_table=block_table,
            request_rows=request_rows,
            token_positions=positions,
            destination_slots=slots,
            miss_mask=miss_mask,
        )

    assert handled
    args = gather.call_args.args
    assert args[0].shape == (layout.hot_blocks, 128, 16)
    assert args[1].shape == (layout.hot_blocks, 128, 16)
    assert args[2] is block_table
    assert args[3].shape == (1, 128, 16)
    assert args[4].shape == (1, 128, 16)
    assert args[5].shape == (2, 1024)
    assert args[6] is request_rows
    assert args[7] is positions
    assert args[8] is slots
    assert args[9] is miss_mask


def test_direct_gather_uses_hot_table_pool_and_returns_original_views() -> None:
    layout = HotCacheLayout(128, 2, 2)
    backend = KVGatherSimBackend(layout)
    kv = torch.ones((layout.hot_blocks, 128, 1, 16), dtype=torch.bfloat16)
    rope = torch.ones((layout.hot_blocks, 128, 1, 16), dtype=torch.bfloat16)
    backend.register_get_cache(
        layer_id=3,
        block_size=128,
        cache_planes=(kv, rope),
    )
    hot_table = layout.block_table(torch.tensor([0, 1], dtype=torch.int32))
    rows = torch.tensor([1, 0], dtype=torch.int32)
    query_start = torch.tensor([0, 2, 3], dtype=torch.int32)
    topk = torch.empty((3, 1, 2048), dtype=torch.int32)
    mapped = torch.empty_like(topk)
    mask = torch.empty_like(topk)

    with patch(
        "vllm_ascend.dsa_offload.kvgather_sim.asu_kv_gather_direct_v2",
        return_value=(
            kv.view(kv.shape[0], 128, -1),
            rope.view(rope.shape[0], 128, -1),
        ),
    ) as gather:
        aliases = backend.gather_history_misses_direct(
            layer_id=3,
            hot_block_table_pool=hot_table,
            request_rows=rows,
            query_start_loc=query_start,
            semantic_topk=topk,
            mapped_indices=mapped,
            gather_mask=mask,
        )

    assert gather.call_args.args[2] is hot_table
    assert gather.call_args.args[6] is rows
    assert gather.call_args.args[7] is query_start
    assert gather.call_args.args[8] is topk
    assert gather.call_args.args[9] is mapped
    assert gather.call_args.args[10] is mask
    assert aliases[0].shape == kv.shape
    assert aliases[1].shape == rope.shape
