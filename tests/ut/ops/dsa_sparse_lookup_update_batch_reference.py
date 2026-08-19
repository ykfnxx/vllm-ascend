# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Deterministic CPU oracle for packed DSA Sparse lookup/update."""

from __future__ import annotations

from collections.abc import Sequence

from tests.ut.ops.dsa_sparse_lookup_update_reference import (
    DSASparseLookupUpdateState,
    INVALID_INDEX,
    _validate_state,
)


def dsa_sparse_lookup_update_batch_reference(
    state: DSASparseLookupUpdateState,
    *,
    req_pool_entries: Sequence[int],
    query_start_loc: Sequence[int],
    query_index: Sequence[Sequence[int]],
    lookup_mask: Sequence[Sequence[int]],
    fallback_slot: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Resolve packed queries while protecting every returned real slot.

    The protected set is request-scoped and survives across all packed query
    rows for that request. Misses that cannot complete an allocate-and-reclaim
    transaction return ``fallback_slot`` without mutating persistent state.
    """

    pool_capacity, index_capacity, slot_count, free_count = (
        _validate_state(state)
    )
    req_num = len(req_pool_entries)
    if req_num == 0:
        raise ValueError("req_pool_entries must not be empty")
    if len(query_start_loc) != req_num + 1:
        raise ValueError("query_start_loc must have req_num + 1 entries")
    if query_start_loc[0] != 0 or query_start_loc[-1] != len(query_index):
        raise ValueError("query_start_loc must cover all packed query rows")
    if any(
        begin > end
        for begin, end in zip(query_start_loc, query_start_loc[1:])
    ):
        raise ValueError("query_start_loc must be monotonic")
    if any(
        begin == end
        for begin, end in zip(query_start_loc, query_start_loc[1:])
    ):
        raise ValueError("every active request must own at least one query")
    if len(set(req_pool_entries)) != req_num:
        raise ValueError("req_pool_entries must be unique in one call")
    if any(not 0 <= row < pool_capacity for row in req_pool_entries):
        raise ValueError("req_pool_entries contains an invalid row")
    if not query_index or not query_index[0]:
        raise ValueError("query_index must be non-empty")
    query_width = len(query_index[0])
    if any(len(row) != query_width for row in query_index):
        raise ValueError("query_index must be rectangular")
    if len(lookup_mask) != len(query_index) or any(
        len(row) != query_width for row in lookup_mask
    ):
        raise ValueError("lookup_mask shape must equal query_index")

    fallback = slot_count if fallback_slot is None else fallback_slot
    slot_out = [[INVALID_INDEX] * query_width for _ in query_index]
    miss_out = [[0] * query_width for _ in query_index]

    for req_id, pool_row in enumerate(req_pool_entries):
        row_index = state.index[pool_row]
        row_slot_to_index = state.slot_to_index[pool_row]
        row_free_slots = state.free_slots[pool_row]
        cursor = state.free_head[pool_row][1]
        protected: set[int] = set()

        for query_id in range(
            query_start_loc[req_id], query_start_loc[req_id + 1]
        ):
            misses: list[tuple[int, int]] = []
            seen_tokens: set[int] = set()
            for entry, token in enumerate(query_index[query_id]):
                if lookup_mask[query_id][entry] == 0:
                    continue
                if not 0 <= token < index_capacity:
                    continue
                if token in seen_tokens:
                    raise ValueError(
                        "active query positions must be unique per query"
                    )
                seen_tokens.add(token)
                slot = row_index[token]
                if slot != INVALID_INDEX:
                    slot_out[query_id][entry] = slot
                    protected.add(slot)
                else:
                    slot_out[query_id][entry] = fallback
                    misses.append((entry, token))

            victims = [
                slot
                for position in range(slot_count)
                if (
                    (slot := (cursor + position) % slot_count)
                    not in protected
                    and row_slot_to_index[slot] != INVALID_INDEX
                )
            ]
            valid_free_slots = [
                slot
                for slot in row_free_slots
                if (
                    0 <= slot < slot_count
                    and row_slot_to_index[slot] == INVALID_INDEX
                )
            ]
            safe_alloc = min(
                len(misses),
                len(valid_free_slots),
                len(victims),
                free_count,
            )

            for miss_rank, (entry, token) in enumerate(
                misses[:safe_alloc]
            ):
                slot = valid_free_slots[miss_rank]
                row_index[token] = slot
                row_slot_to_index[slot] = token
                slot_out[query_id][entry] = slot
                miss_out[query_id][entry] = 1
                protected.add(slot)

            reclaimed = victims[:safe_alloc]
            for victim_rank, slot in enumerate(reclaimed):
                old_token = row_slot_to_index[slot]
                row_slot_to_index[slot] = INVALID_INDEX
                if row_index[old_token] == slot:
                    row_index[old_token] = INVALID_INDEX
                row_free_slots[safe_alloc - 1 - victim_rank] = slot

            if reclaimed:
                cursor = (reclaimed[-1] + 1) % slot_count
                state.free_head[pool_row][1] = cursor
            state.free_head[pool_row][0] = 0

    _validate_state(state)
    return slot_out, miss_out


__all__ = ["dsa_sparse_lookup_update_batch_reference"]
