# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout, HotCacheState
from vllm_ascend.dsa_offload.lookup import (
    DSAOffloadBatch,
    IndexCacheCohort,
    LookupPlan,
    load_plan_misses,
    load_prefetch_misses,
    make_lookup_plan,
    make_prefetch_lookup_plan,
    pack_graph_decode_metadata,
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
        request_rows_cpu=(0,),
        decode_request_indices=(0,),
        query_ranges=((0, 1),),
        query_positions=torch.tensor([8], dtype=torch.int64),
        query_positions_cpu=(8,),
        is_mtp=False,
        committed_block_keys={"decode": [101, 102]},
        candidate_block_keys={},
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
            cohort=cohort,
            batch=batch,
        )

    assert lookup.call_args.args[3].tolist() == [[1, 1, 0, 0]]
    assert plan.mapped_indices.tolist() == [[0, layout.replaceable_base, layout.tail_base, -1]]
    assert plan.miss_positions.tolist() == [5]
    assert plan.miss_logical_blocks.tolist() == [1]
    assert plan.miss_block_offsets.tolist() == [1]
    assert plan.miss_destination_slots.tolist() == [layout.replaceable_base]


def test_graph_plan_keeps_dense_lookup_and_gather_metadata(spy_io) -> None:
    layout = HotCacheLayout(4, 2, 2)
    spy_io.gather_history_misses = lambda **_: True
    cohort = IndexCacheCohort("leader", "leader", ("leader",), (7,))
    state = LookupState(
        index=torch.empty((2, 1), dtype=torch.int32),
        slot_to_index=torch.empty((2, 1), dtype=torch.int32),
        free_slots=torch.empty((2, 1), dtype=torch.int32),
        free_head=torch.empty((2, 1), dtype=torch.int32),
    )
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    batch = DSAOffloadBatch(
        layout=layout,
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(cohort,),
        lookup_states={"leader": state},
        request_ids=("first", "second"),
        request_rows=torch.tensor([1, 0], dtype=torch.int32),
        request_rows_cpu=(1, 0),
        decode_request_indices=(0, 1),
        query_ranges=((0, 2), (2, 3)),
        query_positions=torch.tensor([8, 9, 12], dtype=torch.int64),
        query_positions_cpu=(8, 9, 12),
        is_mtp=False,
        committed_block_keys={"first": [], "second": []},
        candidate_block_keys={},
        graph_query_start_loc=query_start_loc,
        enable_turbo_lookup=True,
    )
    semantic = torch.tensor(
        [[1, 2], [3, 4], [5, 6]],
        dtype=torch.int32,
    )
    slots = torch.zeros_like(semantic)
    misses = torch.ones_like(semantic)

    with patch(
        "vllm_ascend.dsa_offload.lookup.lookup_update",
        return_value=(slots, misses),
    ) as lookup:
        plan = make_lookup_plan(
            semantic_topk=semantic,
            cohort=cohort,
            batch=batch,
        )

    assert lookup.call_args.args[1] is batch.request_rows
    assert torch.equal(lookup.call_args.args[2], semantic)
    assert plan.query_request_rows.tolist() == [1, 1, 0]
    assert plan.query_indices is not None
    assert torch.equal(plan.query_indices, semantic)
    assert plan.lookup_slots is not None
    assert plan.lookup_slots.tolist() == slots.tolist()
    assert plan.dense_miss_mask is not None
    assert plan.dense_miss_mask.tolist() == misses.tolist()
    assert plan.miss_positions.numel() == 0


def test_graph_prefetch_plan_uses_fixed_dense_gather_metadata(spy_io) -> None:
    layout = HotCacheLayout(4, 2, 2)
    gather_calls = []

    def gather_history_misses(**kwargs) -> bool:
        gather_calls.append(kwargs)
        return True

    spy_io.gather_history_misses = gather_history_misses
    cohort = IndexCacheCohort("leader", "leader", ("leader",), (7,))
    state = LookupState(
        index=torch.empty((2, 1), dtype=torch.int32),
        slot_to_index=torch.empty((2, 1), dtype=torch.int32),
        free_slots=torch.empty((2, 1), dtype=torch.int32),
        free_head=torch.empty((2, 1), dtype=torch.int32),
    )
    hot_cache = HotCacheState(
        layout,
        {"leader": (torch.empty((1, 1)),)},
    )
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    batch = DSAOffloadBatch(
        layout=layout,
        hot_cache=hot_cache,
        io_backend=spy_io,
        cohorts=(cohort,),
        lookup_states={"leader": state},
        request_ids=("first", "second"),
        request_rows=torch.tensor([1, -1], dtype=torch.int32),
        request_rows_cpu=(1, -1),
        decode_request_indices=(0, 1),
        query_ranges=((0, 2), (2, 3)),
        query_positions=torch.tensor([8, 9, 12], dtype=torch.int64),
        query_positions_cpu=(8, 9, 12),
        is_mtp=False,
        committed_block_keys={"first": [], "second": []},
        candidate_block_keys={},
        graph_query_start_loc=query_start_loc,
        enable_turbo_prefetch_lookup=True,
    )
    batch.packed_decode = pack_graph_decode_metadata(batch)
    semantic = torch.tensor(
        [[1, 2], [3, 4], [5, 6]],
        dtype=torch.int32,
    )
    slots = torch.zeros_like(semantic)
    misses = torch.ones_like(semantic)

    with patch(
        "vllm_ascend.dsa_offload.lookup.lookup_update",
        return_value=(slots, misses),
    ) as lookup:
        plan = make_prefetch_lookup_plan(
            semantic_topk=semantic,
            cohort=cohort,
            batch=batch,
        )

    assert lookup.call_args.args[1] is batch.request_rows
    assert torch.equal(lookup.call_args.args[2], semantic)
    assert plan.query_request_rows.tolist() == [1, 1, -1]
    assert plan.query_indices is not None
    assert torch.equal(plan.query_indices, semantic)
    assert plan.lookup_slots is not None
    assert plan.dense_miss_mask is not None
    assert plan.dense_miss_mask.tolist() == misses.tolist()
    assert plan.miss_positions.numel() == 0

    load_prefetch_misses(
        plan,
        layer_id=7,
        batch=batch,
        storage_ids=torch.empty((2, 1), dtype=torch.int64),
    )

    assert len(gather_calls) == 1
    assert gather_calls[0]["layer_id"] == 7
    assert gather_calls[0]["request_rows"].tolist() == [1, 1, -1]
    assert torch.equal(
        gather_calls[0]["destination_block_table"],
        layout.block_table(plan.query_request_rows),
    )
    assert torch.equal(gather_calls[0]["token_positions"], semantic)


def test_candidate_keys_extend_committed_prefix(spy_io) -> None:
    batch = DSAOffloadBatch(
        layout=HotCacheLayout(4, 1, 2),
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(),
        lookup_states={},
        request_ids=("request",),
        request_rows=torch.tensor([-1], dtype=torch.int32),
        request_rows_cpu=(-1,),
        decode_request_indices=(),
        query_ranges=((0, 1),),
        query_positions=torch.tensor([0]),
        query_positions_cpu=(0,),
        is_mtp=True,
        committed_block_keys={"request": [101, 102]},
        candidate_block_keys={"request": [103]},
    )

    assert batch.block_keys(0) == (101, 102, 103)


def test_lookup_hit_does_not_call_io(spy_io) -> None:
    batch = DSAOffloadBatch(
        layout=HotCacheLayout(4, 1, 1),
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(),
        lookup_states={},
        request_ids=("request",),
        request_rows=torch.tensor([0], dtype=torch.int32),
        request_rows_cpu=(0,),
        decode_request_indices=(0,),
        query_ranges=((0, 1),),
        query_positions=torch.tensor([0]),
        query_positions_cpu=(0,),
        is_mtp=False,
        committed_block_keys={"request": []},
        candidate_block_keys={},
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
        tail_mask=torch.empty((1, 0), dtype=torch.bool),
        fallback_mask=torch.empty((1, 0), dtype=torch.bool),
        staging_mask=torch.empty((1, 0), dtype=torch.bool),
    )

    load_plan_misses(plan, 0, batch)

    assert spy_io.get_calls == []


def test_lookup_miss_rejects_missing_block_key(spy_io) -> None:
    batch = DSAOffloadBatch(
        layout=HotCacheLayout(4, 1, 1),
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(),
        lookup_states={},
        request_ids=("request",),
        request_rows=torch.tensor([0], dtype=torch.int32),
        request_rows_cpu=(0,),
        decode_request_indices=(0,),
        query_ranges=((0, 1),),
        query_positions=torch.tensor([0]),
        query_positions_cpu=(0,),
        is_mtp=False,
        committed_block_keys={"request": []},
        candidate_block_keys={},
    )
    plan = LookupPlan(
        mapped_indices=torch.empty((1, 0), dtype=torch.int32),
        miss_positions=torch.tensor([5], dtype=torch.int64),
        miss_logical_blocks=torch.tensor([1], dtype=torch.int64),
        miss_block_offsets=torch.tensor([1], dtype=torch.int64),
        miss_destination_slots=torch.tensor([0], dtype=torch.int64),
        miss_batch_indices=torch.tensor([0], dtype=torch.int32),
        query_request_rows=torch.tensor([0], dtype=torch.int32),
        tail_mask=torch.empty((1, 0), dtype=torch.bool),
        fallback_mask=torch.empty((1, 0), dtype=torch.bool),
        staging_mask=torch.empty((1, 0), dtype=torch.bool),
    )

    with pytest.raises(
        RuntimeError,
        match=r"miss load for request request requires 2 block keys",
    ):
        load_plan_misses(plan, 0, batch)
