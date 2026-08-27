# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import accumulate

import torch

from .constants import (
    FALLBACK_SENTINEL,
    FREE_HEAD_STRIDE,
    INDEX_CAPACITY,
    LOOKUP_SLOTS,
    REPLACEABLE_SLOTS,
    RESIDENT_SLOTS,
)
from .hot_cache import HotCacheLayout, HotCacheState
from .io import IOBackend, make_storage_ids
from .ops import LookupState, lookup_update, lookup_update_batch

INVALID_INDEX = -1


@dataclass(frozen=True)
class IndexCacheCohort:
    cohort_id: str
    leader_layer: str
    layer_names: tuple[str, ...]
    layer_ids: tuple[int, ...]


def scan_index_cache_cohorts(
    ordered_layers: Sequence[tuple[str, bool, int]],
) -> tuple[IndexCacheCohort, ...]:
    if not ordered_layers:
        raise ValueError("DSA Offload requires target SFA layers.")
    cohorts: list[IndexCacheCohort] = []
    previous_layer_id = -1
    for layer_name, skip_topk, layer_id in ordered_layers:
        if layer_id <= previous_layer_id:
            raise ValueError("DSA Offload target SFA layers must follow execution order.")
        if skip_topk:
            if not cohorts:
                raise ValueError("An IndexCache follower must follow a cohort leader.")
            cohort = cohorts[-1]
            if layer_id != cohort.layer_ids[-1] + 1:
                raise ValueError("IndexCache followers must be consecutive.")
            cohorts[-1] = IndexCacheCohort(
                cohort_id=cohort.cohort_id,
                leader_layer=cohort.leader_layer,
                layer_names=(*cohort.layer_names, layer_name),
                layer_ids=(*cohort.layer_ids, layer_id),
            )
        else:
            cohorts.append(
                IndexCacheCohort(
                    cohort_id=layer_name,
                    leader_layer=layer_name,
                    layer_names=(layer_name,),
                    layer_ids=(layer_id,),
                )
            )
        previous_layer_id = layer_id
    return tuple(cohorts)


def create_lookup_states(
    cohorts: Sequence[IndexCacheCohort],
    max_num_seqs: int,
    device: torch.device | str,
) -> dict[str, LookupState]:
    return {
        cohort.cohort_id: LookupState(
            index=torch.full(
                (max_num_seqs, INDEX_CAPACITY),
                INVALID_INDEX,
                dtype=torch.int32,
                device=device,
            ),
            slot_to_index=torch.full(
                (max_num_seqs, LOOKUP_SLOTS),
                INVALID_INDEX,
                dtype=torch.int32,
                device=device,
            ),
            free_slots=torch.arange(
                RESIDENT_SLOTS,
                LOOKUP_SLOTS,
                dtype=torch.int32,
                device=device,
            )
            .expand(max_num_seqs, REPLACEABLE_SLOTS)
            .clone(),
            free_head=torch.zeros(
                (max_num_seqs, FREE_HEAD_STRIDE),
                dtype=torch.int32,
                device=device,
            ),
        )
        for cohort in cohorts
    }


def clear_lookup_row(states: Mapping[str, LookupState], row_id: int) -> None:
    for state in states.values():
        state.index[row_id].fill_(INVALID_INDEX)
        state.slot_to_index[row_id].fill_(INVALID_INDEX)
        state.free_slots[row_id].copy_(
            torch.arange(
                RESIDENT_SLOTS,
                LOOKUP_SLOTS,
                dtype=torch.int32,
                device=state.free_slots.device,
            )
        )
        state.free_head[row_id].zero_()


def initialize_resident_mapping(
    state: LookupState,
    row_id: int,
    resident_positions: Sequence[int],
) -> None:
    if not resident_positions:
        return
    positions = torch.tensor(
        resident_positions,
        dtype=torch.int64,
        device=state.index.device,
    )
    slots = torch.arange(
        len(resident_positions),
        dtype=torch.int32,
        device=state.index.device,
    )
    state.index[row_id, positions] = slots
    state.slot_to_index[row_id, : len(resident_positions)] = positions.to(torch.int32)


@dataclass
class LookupPlan:
    mapped_indices: torch.Tensor
    miss_positions: torch.Tensor
    miss_logical_blocks: torch.Tensor
    miss_block_offsets: torch.Tensor
    miss_destination_slots: torch.Tensor
    miss_batch_indices: torch.Tensor
    query_request_rows: torch.Tensor
    hot_block_table: torch.Tensor
    tail_mask: torch.Tensor
    fallback_mask: torch.Tensor
    staging_mask: torch.Tensor


@dataclass
class DSAOffloadBatch:
    layout: HotCacheLayout
    hot_cache: HotCacheState | None
    io_backend: IOBackend
    cohorts: tuple[IndexCacheCohort, ...]
    lookup_states: dict[str, LookupState]
    request_ids: tuple[str, ...]
    request_rows: torch.Tensor
    decode_request_indices: tuple[int, ...]
    query_ranges: tuple[tuple[int, int], ...]
    query_positions: torch.Tensor
    is_mtp: bool
    committed_block_hashes: Mapping[str, Sequence[bytes]]
    candidate_block_hashes: Mapping[str, Sequence[bytes]]
    prefill_state: object | None = None
    lookup_plans: dict[str, LookupPlan] = field(default_factory=dict)

    def block_hashes(self, request_index: int) -> Sequence[bytes]:
        request_id = self.request_ids[request_index]
        committed = self.committed_block_hashes[request_id]
        candidate = self.candidate_block_hashes.get(request_id)
        return committed if candidate is None else (*committed, *candidate)


def build_dsa_offload_batch(
    *,
    layout: HotCacheLayout,
    hot_cache: HotCacheState | None,
    io_backend: IOBackend,
    cohorts: tuple[IndexCacheCohort, ...],
    lookup_states: dict[str, LookupState],
    request_ids: Sequence[str],
    query_counts: Sequence[int],
    query_positions: torch.Tensor,
    is_mtp: bool,
    committed_block_hashes: Mapping[str, Sequence[bytes]],
    candidate_block_hashes: Mapping[str, Sequence[bytes]],
    prefill_state: object | None = None,
) -> DSAOffloadBatch:
    query_ends = tuple(accumulate(query_counts))
    query_ranges = tuple(zip((0, *query_ends[:-1]), query_ends))
    if hot_cache is None:
        row_values = [-1] * len(request_ids)
    else:
        row_values = [hot_cache.request_to_row.get(request_id, -1) for request_id in request_ids]
    request_rows = torch.tensor(
        row_values,
        dtype=torch.int32,
        device=query_positions.device,
    )
    decode_request_indices = tuple(index for index, row_id in enumerate(row_values) if row_id >= 0)
    return DSAOffloadBatch(
        layout=layout,
        hot_cache=hot_cache,
        io_backend=io_backend,
        cohorts=cohorts,
        lookup_states=lookup_states,
        request_ids=tuple(request_ids),
        request_rows=request_rows,
        decode_request_indices=decode_request_indices,
        query_ranges=query_ranges,
        query_positions=query_positions,
        is_mtp=is_mtp,
        committed_block_hashes=committed_block_hashes,
        candidate_block_hashes=candidate_block_hashes,
        prefill_state=prefill_state,
    )


def make_lookup_plan(
    *,
    semantic_topk: torch.Tensor,
    default_block_table: torch.Tensor,
    cohort: IndexCacheCohort,
    batch: DSAOffloadBatch,
) -> LookupPlan:
    topk_shape = semantic_topk.shape
    semantic = semantic_topk.reshape(semantic_topk.shape[0], -1)
    decode_ranges = [batch.query_ranges[index] for index in batch.decode_request_indices]
    packed_topk = torch.cat([semantic[begin:end] for begin, end in decode_ranges], dim=0).to(torch.int32)
    packed_positions = torch.cat([batch.query_positions[begin:end] for begin, end in decode_ranges]).to(torch.int32)
    query_lengths = [end - begin for begin, end in decode_ranges]
    query_start_loc = torch.tensor(
        (0, *accumulate(query_lengths)),
        dtype=torch.int32,
        device=semantic.device,
    )
    request_rows = batch.request_rows[list(batch.decode_request_indices)].contiguous()
    request_indices = torch.tensor(
        batch.decode_request_indices,
        dtype=torch.int32,
        device=semantic.device,
    )
    query_request_rows = torch.repeat_interleave(
        request_rows,
        torch.tensor(query_lengths, dtype=torch.int64, device=semantic.device),
        output_size=packed_topk.shape[0],
    )
    query_batch_indices = torch.repeat_interleave(
        request_indices,
        torch.tensor(query_lengths, dtype=torch.int64, device=semantic.device),
        output_size=packed_topk.shape[0],
    )

    verify_starts = packed_positions[query_start_loc[:-1].long()]
    expanded_verify_starts = torch.repeat_interleave(
        verify_starts,
        torch.tensor(query_lengths, dtype=torch.int64, device=semantic.device),
        output_size=packed_topk.shape[0],
    )
    tail_starts = (
        torch.div(
            expanded_verify_starts,
            batch.layout.block_size,
            rounding_mode="floor",
        )
        * batch.layout.block_size
    )
    current_positions = packed_positions.unsqueeze(1)
    valid_mask = (packed_topk >= 0) & (packed_topk < INDEX_CAPACITY)
    history_mask = valid_mask & (packed_topk < tail_starts.unsqueeze(1))
    tail_mask = (
        valid_mask & (packed_topk >= tail_starts.unsqueeze(1)) & (packed_topk < expanded_verify_starts.unsqueeze(1))
    )
    staging_mask = (
        valid_mask
        & batch.is_mtp
        & (packed_topk >= expanded_verify_starts.unsqueeze(1))
        & (packed_topk <= current_positions)
    )
    if not batch.is_mtp:
        tail_mask = valid_mask & (packed_topk >= tail_starts.unsqueeze(1)) & (packed_topk <= current_positions)

    lookup_mask = history_mask.to(torch.int32).contiguous()
    state = batch.lookup_states[cohort.cohort_id]
    if batch.is_mtp:
        slot_out, miss_out = lookup_update_batch(
            state,
            request_rows,
            query_start_loc,
            packed_topk,
            lookup_mask,
        )
    else:
        slot_out, miss_out = lookup_update(
            state,
            request_rows,
            packed_topk,
            lookup_mask,
        )

    lookup_offsets = batch.layout.lookup_offsets(slot_out)
    fallback_mask = valid_mask & (slot_out == FALLBACK_SENTINEL)
    lookup_offsets = torch.where(
        fallback_mask,
        torch.full_like(lookup_offsets, batch.layout.fallback_slot),
        lookup_offsets,
    )
    tail_offsets = batch.layout.tail_base + packed_topk - tail_starts.unsqueeze(1)
    staging_offsets = batch.layout.staging_base + packed_topk - expanded_verify_starts.unsqueeze(1)
    mapped = torch.where(
        staging_mask,
        staging_offsets,
        torch.where(tail_mask, tail_offsets, lookup_offsets),
    )
    mapped = torch.where(valid_mask, mapped, torch.full_like(mapped, INVALID_INDEX))

    merged_indices = semantic.clone()
    packed_begin = 0
    for request_index, (begin, end) in zip(batch.decode_request_indices, decode_ranges):
        count = end - begin
        merged_indices[begin:end] = mapped[packed_begin : packed_begin + count]
        packed_begin += count

    active_misses = miss_out.bool() & history_mask & ~fallback_mask
    expanded_rows = query_request_rows.unsqueeze(1).expand_as(packed_topk)
    expanded_batch_indices = query_batch_indices.unsqueeze(1).expand_as(packed_topk)
    miss_positions = packed_topk[active_misses].to(torch.int64)
    miss_rows = expanded_rows[active_misses].to(torch.int64)
    miss_slots = batch.layout.lookup_offsets(slot_out[active_misses].to(torch.int64))
    miss_destination_slots = (
        batch.layout.hot_block_base + miss_rows * batch.layout.hot_blocks_per_row
    ) * batch.layout.block_size + miss_slots

    hot_tables = batch.layout.block_table(request_rows)
    merged_block_table = default_block_table.clone()
    for decode_index, request_index in enumerate(batch.decode_request_indices):
        merged_block_table[request_index].zero_()
        merged_block_table[request_index, : hot_tables.shape[1]] = hot_tables[decode_index]

    return LookupPlan(
        mapped_indices=merged_indices.reshape(topk_shape),
        miss_positions=miss_positions,
        miss_logical_blocks=torch.div(
            miss_positions,
            batch.layout.block_size,
            rounding_mode="floor",
        ),
        miss_block_offsets=torch.remainder(miss_positions, batch.layout.block_size),
        miss_destination_slots=miss_destination_slots,
        miss_batch_indices=expanded_batch_indices[active_misses].to(torch.int32),
        query_request_rows=query_request_rows,
        hot_block_table=merged_block_table,
        tail_mask=tail_mask,
        fallback_mask=fallback_mask,
        staging_mask=staging_mask,
    )


def load_plan_misses(
    plan: LookupPlan,
    layer_id: int,
    batch: DSAOffloadBatch,
) -> None:
    if plan.miss_logical_blocks.numel() == 0:
        return
    storage_ids = torch.empty_like(plan.miss_logical_blocks)
    for request_index in batch.decode_request_indices:
        request_mask = plan.miss_batch_indices == request_index
        request_storage_ids = make_storage_ids(
            batch.block_hashes(request_index),
            layer_id,
            device=plan.miss_logical_blocks.device,
        )
        storage_ids[request_mask] = request_storage_ids[plan.miss_logical_blocks[request_mask]]
    batch.io_backend.get_tokens(
        layer_id=layer_id,
        storage_ids=storage_ids,
        token_offsets=plan.miss_block_offsets,
        destination_slots=plan.miss_destination_slots,
    )
