# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

import torch

from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout
from vllm_ascend.dsa_offload.lookup import (
    DSAOffloadBatch,
    IndexCacheCohort,
    LookupPlan,
    load_plan_misses,
    make_lookup_plan,
)
from vllm_ascend.dsa_offload.ops import LookupState


def test_history_tail_and_miss_are_mapped_to_fixed_hot_slots(spy_io) -> None:
    layout = HotCacheLayout(4, 1, 2)
    cohort = IndexCacheCohort("leader", "leader", ("leader",), (7,))
    state = LookupState(
        index=torch.empty((1, 1), dtype=torch.int32),
        slot_to_index=torch.empty((1, 1), dtype=torch.int32),
        free_slots=torch.empty((1, 1), dtype=torch.int32),
        free_head=torch.empty((1, 1), dtype=torch.int32),
    )
    batch = DSAOffloadBatch(
        layout=layout,
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(cohort,),
        lookup_states={"leader": state},
        request_ids=("decode",),
        request_rows=torch.tensor([0], dtype=torch.int32),
        decode_request_indices=(0,),
        query_ranges=((0, 1),),
        query_positions=torch.tensor([8], dtype=torch.int64),
        is_mtp=False,
        committed_block_hashes={"decode": [b"h0", b"h1"]},
        candidate_block_hashes={},
    )
    semantic = torch.tensor([[1, 5, 8, -1]], dtype=torch.int32)
    slots = torch.tensor([[0, 8192, -1, -1]], dtype=torch.int32)
    misses = torch.tensor([[0, 1, 0, 0]], dtype=torch.int32)

    with patch(
        "vllm_ascend.dsa_offload.lookup.lookup_update",
        return_value=(slots, misses),
    ) as lookup:
        plan = make_lookup_plan(
            semantic_topk=semantic,
            default_block_table=torch.full((1, layout.hot_blocks_per_row), 99, dtype=torch.int32),
            cohort=cohort,
            batch=batch,
        )

    assert lookup.call_args.args[3].tolist() == [[1, 1, 0, 0]]
    assert plan.mapped_indices.tolist() == [[0, layout.replaceable_base, layout.tail_base, -1]]
    assert plan.miss_positions.tolist() == [5]
    assert plan.miss_logical_blocks.tolist() == [1]
    assert plan.miss_block_offsets.tolist() == [1]
    assert plan.miss_destination_slots.tolist() == [layout.replaceable_base]
    assert plan.hot_block_table[0, : layout.hot_blocks_per_row].tolist() == list(range(layout.hot_blocks_per_row))


def test_candidate_hashes_extend_committed_prefix(spy_io) -> None:
    batch = DSAOffloadBatch(
        layout=HotCacheLayout(4, 1, 2),
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(),
        lookup_states={},
        request_ids=("request",),
        request_rows=torch.tensor([-1], dtype=torch.int32),
        decode_request_indices=(),
        query_ranges=((0, 1),),
        query_positions=torch.tensor([0]),
        is_mtp=True,
        committed_block_hashes={"request": [b"committed-0", b"committed-1"]},
        candidate_block_hashes={"request": [b"candidate-2"]},
    )

    assert batch.block_hashes(0) == (b"committed-0", b"committed-1", b"candidate-2")


def test_lookup_hit_does_not_call_io(spy_io) -> None:
    batch = DSAOffloadBatch(
        layout=HotCacheLayout(4, 1, 1),
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(),
        lookup_states={},
        request_ids=("request",),
        request_rows=torch.tensor([0], dtype=torch.int32),
        decode_request_indices=(0,),
        query_ranges=((0, 1),),
        query_positions=torch.tensor([0]),
        is_mtp=False,
        committed_block_hashes={"request": []},
        candidate_block_hashes={},
    )
    empty = torch.empty(0, dtype=torch.int64)
    plan = LookupPlan(
        mapped_indices=torch.empty((1, 0), dtype=torch.int32),
        miss_positions=empty,
        miss_logical_blocks=empty,
        miss_block_offsets=empty,
        miss_destination_slots=empty,
        miss_batch_indices=empty.to(torch.int32),
        query_request_rows=torch.tensor([0], dtype=torch.int32),
        hot_block_table=torch.empty((1, 0), dtype=torch.int32),
        tail_mask=torch.empty((1, 0), dtype=torch.bool),
        fallback_mask=torch.empty((1, 0), dtype=torch.bool),
        staging_mask=torch.empty((1, 0), dtype=torch.bool),
    )

    load_plan_misses(plan, 0, batch)

    assert spy_io.get_calls == []
