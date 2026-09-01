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
    get_decode_block_table,
    get_expanded_lookup_boundaries,
    get_packed_addressing_metadata,
    load_plan_misses,
    load_prefetch_misses,
    make_lookup_plan,
    make_prefetch_lookup_plan,
    pack_graph_decode_metadata,
)
from vllm_ascend.dsa_offload.ops import LookupState


def test_packed_addressing_metadata_is_shared_within_decode_step(
    spy_io,
) -> None:
    layout = HotCacheLayout(4, 2, 2)
    batch = DSAOffloadBatch(
        layout=layout,
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(),
        lookup_states={},
        request_ids=("first", "second"),
        request_rows=torch.tensor([1, 0], dtype=torch.int32),
        decode_request_indices=(0, 1),
        query_ranges=((0, 2), (2, 3)),
        query_positions=torch.tensor([7, 8, 12], dtype=torch.int64),
        is_mtp=True,
        committed_block_hashes={"first": [], "second": []},
        candidate_block_hashes={},
        graph_query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
    )

    first = get_packed_addressing_metadata(batch)
    second = get_packed_addressing_metadata(batch)
    expanded_verify, expanded_tail = get_expanded_lookup_boundaries(batch)
    reused_verify, reused_tail = get_expanded_lookup_boundaries(batch)

    assert second is first
    assert first.query_lengths.tolist() == [2, 1]
    assert first.query_request_rows.tolist() == [1, 1, 0]
    assert first.query_positions.tolist() == [7, 8, 12]
    assert first.cumulative_query_lengths.tolist() == [2, 3]
    assert first.verify_starts.tolist() == [7, 12]
    assert first.tail_starts.tolist() == [4, 12]
    assert expanded_verify.tolist() == [7, 7, 12]
    assert expanded_tail.tolist() == [4, 4, 12]
    assert reused_verify is expanded_verify
    assert reused_tail is expanded_tail
    assert first.expanded_query_starts is not None
    assert first.expanded_query_starts.tolist() == [0, 0, 2]


def test_new_decode_batch_recomputes_growing_sequence_metadata(spy_io) -> None:
    layout = HotCacheLayout(4, 1, 2)

    def make_batch(position: int) -> DSAOffloadBatch:
        return DSAOffloadBatch(
            layout=layout,
            hot_cache=None,
            io_backend=spy_io,
            cohorts=(),
            lookup_states={},
            request_ids=("request",),
            request_rows=torch.tensor([0], dtype=torch.int32),
            decode_request_indices=(0,),
            query_ranges=((0, 1),),
            query_positions=torch.tensor([position], dtype=torch.int64),
            is_mtp=False,
            committed_block_hashes={"request": []},
            candidate_block_hashes={},
        )

    previous = get_packed_addressing_metadata(make_batch(7))
    current = get_packed_addressing_metadata(make_batch(8))

    assert previous.verify_starts.tolist() == [7]
    assert previous.tail_starts.tolist() == [4]
    assert current.verify_starts.tolist() == [8]
    assert current.tail_starts.tolist() == [8]


def test_decode_block_table_reuse_is_isolated_by_metadata_source(
    spy_io,
) -> None:
    batch = DSAOffloadBatch(
        layout=HotCacheLayout(4, 2, 2),
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(),
        lookup_states={},
        request_ids=("prefill", "decode"),
        request_rows=torch.tensor([-1, 0], dtype=torch.int32),
        decode_request_indices=(1,),
        query_ranges=((0, 1), (1, 2)),
        query_positions=torch.tensor([0, 8], dtype=torch.int64),
        is_mtp=False,
        committed_block_hashes={"prefill": [], "decode": []},
        candidate_block_hashes={},
    )
    first_source = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    second_source = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)

    first = get_decode_block_table(batch, first_source)
    reused = get_decode_block_table(batch, first_source)
    second = get_decode_block_table(batch, second_source)

    assert reused is first
    assert first.tolist() == [[3, 4]]
    assert second.tolist() == [[7, 8]]
    assert second is not first


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
        decode_request_indices=(0, 1),
        query_ranges=((0, 2), (2, 3)),
        query_positions=torch.tensor([8, 9, 12], dtype=torch.int64),
        is_mtp=False,
        committed_block_hashes={"first": [], "second": []},
        candidate_block_hashes={},
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
    assert batch.packed_addressing is not None
    assert plan.query_request_rows is batch.packed_addressing.query_request_rows
    assert plan.query_indices is not None
    assert torch.equal(plan.query_indices, semantic)
    assert plan.lookup_slots is not None
    assert plan.lookup_slots.tolist() == slots.tolist()
    assert plan.dense_miss_mask is not None
    assert plan.dense_miss_mask.tolist() == misses.tolist()
    assert plan.miss_positions.numel() == 0


def test_mtp_fused_lookup_consumes_shared_compact_metadata(spy_io) -> None:
    layout = HotCacheLayout(4, 2, 2)
    spy_io.gather_history_misses = lambda **_: True
    cohort = IndexCacheCohort("leader", "leader", ("leader",), (7,))
    state = LookupState(
        index=torch.empty((2, 1), dtype=torch.int32),
        slot_to_index=torch.empty((2, 1), dtype=torch.int32),
        free_slots=torch.empty((2, 1), dtype=torch.int32),
        free_head=torch.empty((2, 1), dtype=torch.int32),
    )
    batch = DSAOffloadBatch(
        layout=layout,
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(cohort,),
        lookup_states={"leader": state},
        request_ids=("first", "second"),
        request_rows=torch.tensor([1, 0], dtype=torch.int32),
        decode_request_indices=(0, 1),
        query_ranges=((0, 2), (2, 3)),
        query_positions=torch.tensor([8, 9, 12], dtype=torch.int64),
        is_mtp=True,
        committed_block_hashes={"first": [], "second": []},
        candidate_block_hashes={},
        graph_query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        enable_turbo_lookup=True,
        enable_turbo_fused_lookup=True,
    )
    semantic = torch.tensor(
        [[1, 2], [3, 4], [5, 6]],
        dtype=torch.int32,
    )
    mapped = torch.tensor(
        [[20, 21], [22, 23], [24, 25]],
        dtype=torch.int32,
    )
    misses = torch.tensor(
        [[1, 0], [0, 1], [1, 0]],
        dtype=torch.int32,
    )

    with patch(
        "vllm_ascend.dsa_offload.lookup.turbo_fused_lookup_update_batch",
        return_value=(mapped, misses),
    ) as fused:
        plan = make_lookup_plan(
            semantic_topk=semantic,
            cohort=cohort,
            batch=batch,
        )

    addressing = batch.packed_addressing
    assert addressing is not None
    assert addressing.expanded_verify_starts is None
    assert addressing.expanded_tail_starts is None
    assert batch.packed_decode is not None
    assert fused.call_args.args[0] is state
    assert fused.call_args.args[1] is batch.request_rows
    assert fused.call_args.args[2] is batch.packed_decode.query_start_loc
    assert torch.equal(fused.call_args.args[3], semantic)
    assert fused.call_args.args[4] is addressing.query_positions
    assert fused.call_args.args[5] is addressing.verify_starts
    assert fused.call_args.args[6] is addressing.tail_starts
    assert fused.call_args.args[7:] == (layout.block_size, 1)
    assert torch.equal(plan.mapped_indices, mapped)
    assert plan.lookup_slots is mapped
    assert plan.dense_miss_mask is misses


def test_mtp_fused_lookup_keeps_mapped_slot_in_eager_miss_plan(spy_io) -> None:
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
        is_mtp=True,
        committed_block_hashes={"decode": []},
        candidate_block_hashes={},
        enable_turbo_lookup=True,
        enable_turbo_fused_lookup=True,
    )
    semantic = torch.tensor([[1]], dtype=torch.int32)
    mapped = torch.tensor(
        [[layout.replaceable_base]],
        dtype=torch.int32,
    )
    misses = torch.ones_like(mapped)

    with patch(
        "vllm_ascend.dsa_offload.lookup.turbo_fused_lookup_update_batch",
        return_value=(mapped, misses),
    ):
        plan = make_lookup_plan(
            semantic_topk=semantic,
            cohort=cohort,
            batch=batch,
        )

    assert plan.miss_positions.tolist() == [1]
    assert plan.miss_destination_slots.tolist() == [
        layout.replaceable_base
    ]


def test_fused_lookup_flag_does_not_bypass_disabled_turbo(spy_io) -> None:
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
        is_mtp=True,
        committed_block_hashes={"decode": []},
        candidate_block_hashes={},
        enable_turbo_lookup=False,
        enable_turbo_fused_lookup=True,
    )
    semantic = torch.tensor([[1]], dtype=torch.int32)
    slots = torch.zeros_like(semantic)
    misses = torch.zeros_like(semantic)

    with (
        patch(
            "vllm_ascend.dsa_offload.lookup.lookup_update_batch",
            return_value=(slots, misses),
        ) as batch_lookup,
        patch(
            "vllm_ascend.dsa_offload.lookup."
            "turbo_fused_lookup_update_batch",
        ) as fused_lookup,
    ):
        make_lookup_plan(
            semantic_topk=semantic,
            cohort=cohort,
            batch=batch,
        )

    batch_lookup.assert_called_once()
    fused_lookup.assert_not_called()


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
        decode_request_indices=(0, 1),
        query_ranges=((0, 2), (2, 3)),
        query_positions=torch.tensor([8, 9, 12], dtype=torch.int64),
        is_mtp=False,
        committed_block_hashes={"first": [], "second": []},
        candidate_block_hashes={},
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
    assert batch.packed_addressing is not None
    assert plan.query_request_rows is batch.packed_addressing.query_request_rows
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
    assert batch.packed_addressing is not None
    assert (
        gather_calls[0]["destination_block_table"]
        is batch.packed_addressing.gather_destination_block_table
    )
    assert torch.equal(
        gather_calls[0]["destination_block_table"],
        layout.block_table(plan.query_request_rows),
    )
    assert torch.equal(gather_calls[0]["token_positions"], semantic)


def test_mtp_fused_prefetch_consumes_only_shared_tail_anchor(spy_io) -> None:
    layout = HotCacheLayout(4, 2, 2)
    spy_io.gather_history_misses = lambda **_: True
    cohort = IndexCacheCohort("leader", "leader", ("leader",), (7,))
    state = LookupState(
        index=torch.empty((2, 1), dtype=torch.int32),
        slot_to_index=torch.empty((2, 1), dtype=torch.int32),
        free_slots=torch.empty((2, 1), dtype=torch.int32),
        free_head=torch.empty((2, 1), dtype=torch.int32),
    )
    batch = DSAOffloadBatch(
        layout=layout,
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(cohort,),
        lookup_states={"leader": state},
        request_ids=("first", "second"),
        request_rows=torch.tensor([1, 0], dtype=torch.int32),
        decode_request_indices=(0, 1),
        query_ranges=((0, 2), (2, 3)),
        query_positions=torch.tensor([8, 9, 12], dtype=torch.int64),
        is_mtp=True,
        committed_block_hashes={"first": [], "second": []},
        candidate_block_hashes={},
        graph_query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        enable_turbo_prefetch_lookup=True,
        enable_turbo_fused_prefetch_lookup=True,
    )
    semantic = torch.tensor(
        [[1, 2], [3, 4], [5, 6]],
        dtype=torch.int32,
    )
    logical_slots = torch.tensor(
        [[20, 21], [22, 23], [24, 25]],
        dtype=torch.int32,
    )
    misses = torch.ones_like(logical_slots)

    with patch(
        "vllm_ascend.dsa_offload.lookup."
        "turbo_fused_prefetch_lookup_update_batch",
        return_value=(logical_slots, misses),
    ) as fused:
        plan = make_prefetch_lookup_plan(
            semantic_topk=semantic,
            cohort=cohort,
            batch=batch,
        )

    addressing = batch.packed_addressing
    assert addressing is not None
    assert addressing.expanded_verify_starts is None
    assert addressing.expanded_tail_starts is None
    assert batch.packed_decode is not None
    assert fused.call_args.args[0] is state
    assert fused.call_args.args[1] is batch.request_rows
    assert fused.call_args.args[2] is batch.packed_decode.query_start_loc
    assert torch.equal(fused.call_args.args[3], semantic)
    assert fused.call_args.args[4] is addressing.tail_starts
    assert fused.call_args.args[5] == layout.block_size
    assert plan.lookup_slots is logical_slots
    assert plan.dense_miss_mask is misses


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
    )

    load_plan_misses(plan, 0, batch)

    assert spy_io.get_calls == []


def test_lookup_miss_rejects_missing_block_hash(spy_io) -> None:
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
    plan = LookupPlan(
        mapped_indices=torch.empty((1, 0), dtype=torch.int32),
        miss_positions=torch.tensor([5], dtype=torch.int64),
        miss_logical_blocks=torch.tensor([1], dtype=torch.int64),
        miss_block_offsets=torch.tensor([1], dtype=torch.int64),
        miss_destination_slots=torch.tensor([0], dtype=torch.int64),
        miss_batch_indices=torch.tensor([0], dtype=torch.int32),
        query_request_rows=torch.tensor([0], dtype=torch.int32),
    )

    with pytest.raises(
        RuntimeError,
        match=r"miss load for request request requires 2 block hashes",
    ):
        load_plan_misses(plan, 0, batch)
