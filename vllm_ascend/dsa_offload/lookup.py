# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import TYPE_CHECKING

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
from .io import IOBackend, make_storage_ids, require_block_hashes
from .ops import (
    LookupState,
    lookup_update,
    lookup_update_batch,
    turbo_lookup_update_batch,
    turbo_prefetch_lookup_update_batch,
    turbo_fused_lookup_update_batch,
    turbo_fused_prefetch_lookup_update_batch,
)

if TYPE_CHECKING:
    from .sfa import SFAAddressingWorkspace

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
    tail_mask: torch.Tensor
    fallback_mask: torch.Tensor
    staging_mask: torch.Tensor
    query_indices: torch.Tensor | None = None
    lookup_slots: torch.Tensor | None = None
    dense_miss_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class PackedDecodeMetadata:
    request_indices: torch.Tensor
    request_rows: torch.Tensor
    token_indices: torch.Tensor
    query_start_loc: torch.Tensor
    query_positions: torch.Tensor
    query_request_rows: torch.Tensor
    query_batch_indices: torch.Tensor


@dataclass(frozen=True)
class PrefetchLookupPlan:
    query_indices: torch.Tensor | None
    lookup_slots: torch.Tensor | None
    dense_miss_mask: torch.Tensor | None
    query_request_rows: torch.Tensor
    miss_positions: torch.Tensor
    miss_logical_blocks: torch.Tensor
    miss_block_offsets: torch.Tensor
    miss_destination_slots: torch.Tensor
    miss_request_rows: torch.Tensor


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
    sfa_workspace: "SFAAddressingWorkspace | None" = None
    decode_request_indices_tensor: torch.Tensor | None = None
    graph_query_start_loc: torch.Tensor | None = None
    packed_decode: PackedDecodeMetadata | None = None
    prefetch_runtime: object | None = None
    enable_turbo_lookup: bool = False
    enable_turbo_prefetch_lookup: bool = False
    enable_turbo_fused_lookup: bool = False
    enable_turbo_fused_prefetch_lookup: bool = False
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
    sfa_workspace: "SFAAddressingWorkspace | None" = None,
    prefetch_runtime: object | None = None,
    enable_turbo_lookup: bool = False,
    enable_turbo_prefetch_lookup: bool = False,
    enable_turbo_fused_lookup: bool = False,
    enable_turbo_fused_prefetch_lookup: bool = False,
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
    batch = DSAOffloadBatch(
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
        sfa_workspace=sfa_workspace,
        decode_request_indices_tensor=torch.tensor(
            decode_request_indices,
            dtype=torch.int64,
            device=query_positions.device,
        ),
        prefetch_runtime=prefetch_runtime,
        enable_turbo_lookup=enable_turbo_lookup,
        enable_turbo_prefetch_lookup=enable_turbo_prefetch_lookup,
        enable_turbo_fused_lookup=enable_turbo_fused_lookup,
        enable_turbo_fused_prefetch_lookup=enable_turbo_fused_prefetch_lookup,
    )
    batch.packed_decode = pack_decode_metadata(batch)
    return batch


def pack_decode_metadata(batch: DSAOffloadBatch) -> PackedDecodeMetadata:
    device = batch.query_positions.device
    request_indices = torch.tensor(
        batch.decode_request_indices,
        dtype=torch.int32,
        device=device,
    )
    request_rows = batch.request_rows.index_select(
        0,
        request_indices.to(torch.int64),
    ).contiguous()
    query_lengths = [
        batch.query_ranges[index][1] - batch.query_ranges[index][0]
        for index in batch.decode_request_indices
    ]
    token_indices = torch.tensor(
        [
            token_index
            for request_index in batch.decode_request_indices
            for token_index in range(*batch.query_ranges[request_index])
        ],
        dtype=torch.int64,
        device=device,
    )
    query_start_loc = torch.tensor(
        (0, *accumulate(query_lengths)),
        dtype=torch.int32,
        device=device,
    )
    query_positions = batch.query_positions.index_select(
        0,
        token_indices,
    ).to(torch.int32)
    if token_indices.numel() == 0:
        query_request_rows = request_rows.new_empty((0,))
        query_batch_indices = request_indices.new_empty((0,))
    else:
        repeats = torch.tensor(
            query_lengths,
            dtype=torch.int64,
            device=device,
        )
        query_request_rows = torch.repeat_interleave(
            request_rows,
            repeats,
            output_size=token_indices.shape[0],
        )
        query_batch_indices = torch.repeat_interleave(
            request_indices,
            repeats,
            output_size=token_indices.shape[0],
        )
    return PackedDecodeMetadata(
        request_indices=request_indices,
        request_rows=request_rows,
        token_indices=token_indices,
        query_start_loc=query_start_loc,
        query_positions=query_positions,
        query_request_rows=query_request_rows,
        query_batch_indices=query_batch_indices,
    )


def pack_graph_decode_metadata(batch: DSAOffloadBatch) -> PackedDecodeMetadata:
    """Build fixed-address Decode metadata for one full-graph capture.

    The dynamic request rows, query starts and positions remain views of the
    model-runner buffers that are updated before replay. Only graph-shape
    indices are allocated here. ``query_request_rows`` is intentionally empty:
    graph lookup plans derive it inside capture so replay observes the current
    request-row values instead of values copied while the graph was built.
    """

    query_start_loc = batch.graph_query_start_loc
    if query_start_loc is None:
        raise ValueError("Graph Decode metadata requires graph_query_start_loc.")
    device = batch.query_positions.device
    num_requests = batch.request_rows.shape[0]
    num_tokens = batch.query_positions.shape[0]
    request_indices = torch.arange(
        num_requests,
        dtype=torch.int32,
        device=device,
    )
    token_indices = torch.arange(
        num_tokens,
        dtype=torch.int64,
        device=device,
    )
    return PackedDecodeMetadata(
        request_indices=request_indices,
        request_rows=batch.request_rows,
        token_indices=token_indices,
        query_start_loc=query_start_loc,
        query_positions=batch.query_positions,
        query_request_rows=batch.request_rows.new_empty((0,)),
        query_batch_indices=request_indices.new_empty((0,)),
    )


def make_lookup_plan(
    *,
    semantic_topk: torch.Tensor,
    cohort: IndexCacheCohort,
    batch: DSAOffloadBatch,
) -> LookupPlan:
    topk_shape = semantic_topk.shape
    semantic = semantic_topk.reshape(semantic_topk.shape[0], -1)
    graph_mode = batch.graph_query_start_loc is not None
    use_dense_gather = hasattr(batch.io_backend, "gather_history_misses")
    if graph_mode:
        packed_topk = semantic.to(torch.int32).contiguous()
        packed_positions = batch.query_positions.to(torch.int32).contiguous()
        query_start_loc = batch.graph_query_start_loc
        request_rows = batch.request_rows
        query_lengths_tensor = (
            query_start_loc[1:] - query_start_loc[:-1]
        ).to(torch.int64)
        query_request_rows = torch.repeat_interleave(
            request_rows,
            query_lengths_tensor,
            output_size=packed_topk.shape[0],
        )
        query_batch_indices = None
    else:
        packed = batch.packed_decode or pack_decode_metadata(batch)
        packed_topk = semantic.index_select(0, packed.token_indices).to(
            torch.int32
        )
        packed_positions = packed.query_positions
        query_start_loc = packed.query_start_loc
        query_lengths_tensor = (
            query_start_loc[1:] - query_start_loc[:-1]
        ).to(torch.int64)
        request_rows = packed.request_rows
        query_request_rows = packed.query_request_rows
        query_batch_indices = packed.query_batch_indices

    verify_starts = packed_positions[query_start_loc[:-1].long()]
    state = batch.lookup_states[cohort.cohort_id]
    if batch.is_mtp and batch.enable_turbo_fused_lookup:
        # Stage-3 3B fused op: in-kernel classification + address mapping.
        # Outputs the final mapped indices and the dense Gather miss mask
        # directly (no where chain, no [T,K] lookup_mask round trip).
        mapped, fused_dense_miss = turbo_fused_lookup_update_batch(
            state,
            request_rows,
            query_start_loc,
            packed_topk,
            packed_positions,
            verify_starts,
            batch.layout.block_size,
            int(batch.is_mtp),
        )
        # The fused op classifies in-kernel; the framework-level
        # tail/fallback/staging masks have no runtime consumers and are kept
        # empty (see test-feedback-note.md stage 3B).  The bool conversion is
        # only needed by the eager (non-graph) sparse-miss indexing path.
        tail_mask = packed_topk.new_empty((0,))
        fallback_mask = packed_topk.new_empty((0,))
        staging_mask = packed_topk.new_empty((0,))
        active_misses = (
            None if graph_mode else fused_dense_miss.bool()
        )
        if graph_mode:
            merged_indices = mapped
        else:
            merged_indices = semantic.clone()
            merged_indices.index_copy_(0, packed.token_indices, mapped)
    else:
        expanded_verify_starts = torch.repeat_interleave(
            verify_starts,
            query_lengths_tensor,
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
        if batch.is_mtp:
            lookup = (
                turbo_lookup_update_batch
                if batch.enable_turbo_lookup
                else lookup_update_batch
            )
            slot_out, miss_out = lookup(
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

        if graph_mode:
            merged_indices = mapped
        else:
            merged_indices = semantic.clone()
            merged_indices.index_copy_(0, packed.token_indices, mapped)

        active_misses = miss_out.bool() & history_mask & ~fallback_mask
    if graph_mode:
        miss_positions = packed_topk.new_empty((0,), dtype=torch.int64)
        miss_logical_blocks = packed_topk.new_empty((0,), dtype=torch.int64)
        miss_block_offsets = packed_topk.new_empty((0,), dtype=torch.int64)
        miss_destination_slots = packed_topk.new_empty((0,), dtype=torch.int64)
        miss_batch_indices = packed_topk.new_empty((0,), dtype=torch.int32)
    else:
        assert query_batch_indices is not None
        expanded_rows = query_request_rows.unsqueeze(1).expand_as(packed_topk)
        expanded_batch_indices = query_batch_indices.unsqueeze(1).expand_as(
            packed_topk
        )
        miss_positions = packed_topk[active_misses].to(torch.int64)
        miss_logical_blocks = torch.div(
            miss_positions,
            batch.layout.block_size,
            rounding_mode="floor",
        )
        miss_block_offsets = torch.remainder(
            miss_positions, batch.layout.block_size
        )
        miss_rows = expanded_rows[active_misses].to(torch.int64)
        miss_slots = batch.layout.lookup_offsets(
            slot_out[active_misses].to(torch.int64)
        )
        miss_destination_slots = (
            batch.layout.hot_block_base
            + miss_rows * batch.layout.hot_blocks_per_row
        ) * batch.layout.block_size + miss_slots
        miss_batch_indices = expanded_batch_indices[active_misses].to(
            torch.int32
        )

    return LookupPlan(
        mapped_indices=merged_indices.reshape(topk_shape),
        query_indices=packed_topk if use_dense_gather else None,
        lookup_slots=(
            mapped
            if (use_dense_gather and batch.enable_turbo_fused_lookup and batch.is_mtp)
            else (
                batch.layout.lookup_offsets(slot_out)
                if use_dense_gather
                else None
            )
        ),
        dense_miss_mask=(
            fused_dense_miss
            if (use_dense_gather and batch.enable_turbo_fused_lookup and batch.is_mtp)
            else (
                active_misses.to(torch.int32).contiguous()
                if use_dense_gather
                else None
            )
        ),
        miss_positions=miss_positions,
        miss_logical_blocks=miss_logical_blocks,
        miss_block_offsets=miss_block_offsets,
        miss_destination_slots=miss_destination_slots,
        miss_batch_indices=miss_batch_indices,
        query_request_rows=query_request_rows,
        tail_mask=tail_mask,
        fallback_mask=fallback_mask,
        staging_mask=staging_mask,
    )


def make_prefetch_lookup_plan(
    *,
    semantic_topk: torch.Tensor,
    cohort: IndexCacheCohort,
    batch: DSAOffloadBatch,
) -> PrefetchLookupPlan:
    packed = batch.packed_decode or pack_decode_metadata(batch)
    graph_mode = batch.graph_query_start_loc is not None
    use_dense_gather = hasattr(batch.io_backend, "gather_history_misses")
    query_indices = semantic_topk.reshape(semantic_topk.shape[0], -1).to(
        torch.int32
    )
    if query_indices.shape[0] != packed.token_indices.shape[0]:
        raise ValueError(
            "DSA Offload prefetch Top-K rows must match packed Decode queries."
        )
    query_start_loc = (
        batch.graph_query_start_loc
        if graph_mode
        else packed.query_start_loc
    )
    assert query_start_loc is not None
    packed_positions = (
        batch.query_positions.to(torch.int32)
        if graph_mode
        else packed.query_positions
    )
    verify_starts = packed_positions[
        query_start_loc[:-1].to(torch.int64)
    ]
    query_lengths = (
        query_start_loc[1:] - query_start_loc[:-1]
    ).to(torch.int64)
    state = batch.lookup_states[cohort.cohort_id]
    request_rows = batch.request_rows if graph_mode else packed.request_rows
    query_request_rows = torch.repeat_interleave(
        request_rows,
        query_lengths,
        output_size=query_indices.shape[0],
    )
    if batch.is_mtp:
        if batch.enable_turbo_fused_prefetch_lookup:
            # Stage-3 3B fused prefetch op: in-kernel history classification
            # (valid && token < tail_start) and logical-slot mapping; outputs
            # lookup_slots (plan.lookup_slots semantics) + dense miss mask
            # directly.
            lookup_slots_fused, miss_out = (
                turbo_fused_prefetch_lookup_update_batch(
                    state,
                    request_rows,
                    query_start_loc,
                    query_indices,
                    packed_positions,
                    verify_starts,
                    batch.layout.block_size,
                )
            )
            fused_prefetch = True
        else:
            expanded_verify_starts = torch.repeat_interleave(
                verify_starts,
                query_lengths,
                output_size=query_indices.shape[0],
            )
            tail_starts = (
                torch.div(
                    expanded_verify_starts,
                    batch.layout.block_size,
                    rounding_mode="floor",
                )
                * batch.layout.block_size
            )
            valid_mask = (query_indices >= 0) & (query_indices < INDEX_CAPACITY)
            lookup_mask = (
                valid_mask & (query_indices < tail_starts.unsqueeze(1))
            ).to(torch.int32).contiguous()
            lookup = (
                turbo_prefetch_lookup_update_batch
                if batch.enable_turbo_prefetch_lookup
                else lookup_update_batch
            )
            slot_out, miss_out = lookup(
                state,
                request_rows,
                query_start_loc,
                query_indices,
                lookup_mask,
            )
            fused_prefetch = False
    else:
        expanded_verify_starts = torch.repeat_interleave(
            verify_starts,
            query_lengths,
            output_size=query_indices.shape[0],
        )
        tail_starts = (
            torch.div(
                expanded_verify_starts,
                batch.layout.block_size,
                rounding_mode="floor",
            )
            * batch.layout.block_size
        )
        valid_mask = (query_indices >= 0) & (query_indices < INDEX_CAPACITY)
        lookup_mask = (
            valid_mask & (query_indices < tail_starts.unsqueeze(1))
        ).to(torch.int32).contiguous()
        slot_out, miss_out = lookup_update(
            state,
            request_rows,
            query_indices,
            lookup_mask,
        )
        fused_prefetch = False

    if fused_prefetch:
        # miss_mask semantics = active (history miss allocated, slot valid)
        active = miss_out.bool()
        slot_out = lookup_slots_fused
    else:
        active = (
            miss_out.bool()
            & lookup_mask.bool()
            & (slot_out >= 0)
            & (slot_out < LOOKUP_SLOTS)
        )
    if graph_mode or use_dense_gather:
        miss_positions = query_indices.new_empty((0,), dtype=torch.int64)
        miss_logical_blocks = query_indices.new_empty((0,), dtype=torch.int64)
        miss_block_offsets = query_indices.new_empty((0,), dtype=torch.int64)
        miss_destination_slots = query_indices.new_empty((0,), dtype=torch.int64)
        miss_rows = query_indices.new_empty((0,), dtype=torch.int64)
    else:
        expanded_rows = query_request_rows.unsqueeze(1).expand_as(
            query_indices
        )
        miss_positions = query_indices[active].to(torch.int64)
        miss_rows = expanded_rows[active].to(torch.int64)
        miss_slots = batch.layout.lookup_offsets(
            slot_out[active].to(torch.int64)
        )
        miss_destination_slots = (
            batch.layout.hot_block_base
            + miss_rows * batch.layout.hot_blocks_per_row
        ) * batch.layout.block_size + miss_slots
        miss_logical_blocks = torch.div(
            miss_positions,
            batch.layout.block_size,
            rounding_mode="floor",
        )
        miss_block_offsets = torch.remainder(
            miss_positions,
            batch.layout.block_size,
        )
    return PrefetchLookupPlan(
        query_indices=query_indices if use_dense_gather else None,
        lookup_slots=(
            slot_out
            if (use_dense_gather and fused_prefetch)
            else (
                batch.layout.lookup_offsets(slot_out)
                if use_dense_gather
                else None
            )
        ),
        dense_miss_mask=(
            active.to(torch.int32).contiguous()
            if use_dense_gather
            else None
        ),
        query_request_rows=query_request_rows,
        miss_positions=miss_positions,
        miss_logical_blocks=miss_logical_blocks,
        miss_block_offsets=miss_block_offsets,
        miss_destination_slots=miss_destination_slots,
        miss_request_rows=miss_rows,
    )


def load_prefetch_misses(
    plan: PrefetchLookupPlan,
    layer_id: int,
    batch: DSAOffloadBatch,
    storage_ids: torch.Tensor,
) -> None:
    dense_gather = getattr(batch.io_backend, "gather_history_misses", None)
    if dense_gather is not None:
        if (
            plan.query_indices is None
            or plan.lookup_slots is None
            or plan.dense_miss_mask is None
        ):
            raise RuntimeError(
                "DSA Offload dense prefetch Gather metadata is unavailable"
            )
        hot_cache = batch.hot_cache
        if hot_cache is None:
            raise RuntimeError(
                "DSA Offload dense prefetch Gather requires a Hot Cache"
            )
        destination_block_table = hot_cache.layout.block_table(
            plan.query_request_rows
        )
        if dense_gather(
            layer_id=layer_id,
            destination_block_table=destination_block_table,
            request_rows=plan.query_request_rows,
            token_positions=plan.query_indices,
            destination_slots=plan.lookup_slots,
            miss_mask=plan.dense_miss_mask,
        ):
            return
    if plan.miss_positions.numel() == 0:
        return
    batch.io_backend.get_tokens(
        layer_id=layer_id,
        storage_ids=storage_ids[
            plan.miss_request_rows,
            plan.miss_logical_blocks,
        ].contiguous(),
        token_offsets=plan.miss_block_offsets.contiguous(),
        destination_slots=plan.miss_destination_slots.contiguous(),
    )


def load_plan_misses(
    plan: LookupPlan,
    layer_id: int,
    batch: DSAOffloadBatch,
) -> None:
    dense_gather = getattr(batch.io_backend, "gather_history_misses", None)
    if dense_gather is not None:
        if (
            plan.query_indices is None
            or plan.lookup_slots is None
            or plan.dense_miss_mask is None
        ):
            raise RuntimeError("DSA Offload dense gather metadata is unavailable")
        hot_cache = batch.hot_cache
        if hot_cache is None:
            raise RuntimeError("DSA Offload dense gather requires a Hot Cache")
        destination_block_table = hot_cache.layout.block_table(
            plan.query_request_rows
        )
        if dense_gather(
            layer_id=layer_id,
            destination_block_table=destination_block_table,
            request_rows=plan.query_request_rows,
            token_positions=plan.query_indices,
            destination_slots=plan.lookup_slots,
            miss_mask=plan.dense_miss_mask,
        ):
            return
    if plan.miss_logical_blocks.numel() == 0:
        return
    storage_ids = torch.empty_like(plan.miss_logical_blocks)
    for request_index in batch.decode_request_indices:
        request_mask = plan.miss_batch_indices == request_index
        request_logical_blocks = plan.miss_logical_blocks[request_mask]
        if request_logical_blocks.numel() == 0:
            continue
        request_id = batch.request_ids[request_index]
        block_hashes = batch.block_hashes(request_index)
        require_block_hashes(
            block_hashes,
            int(request_logical_blocks.max().item()) + 1,
            context=f"miss load for request {request_id}",
        )
        request_storage_ids = make_storage_ids(
            block_hashes,
            layer_id,
            device=plan.miss_logical_blocks.device,
        )
        storage_ids[request_mask] = request_storage_ids[
            request_logical_blocks
        ]
    batch.io_backend.get_tokens(
        layer_id=layer_id,
        storage_ids=storage_ids,
        token_offsets=plan.miss_block_offsets,
        destination_slots=plan.miss_destination_slots,
    )
