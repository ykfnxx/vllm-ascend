# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Deterministic CPU oracle for fused ASU-shaped lookup and maintain."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


INVALID_INDEX = -1


@dataclass
class DSASparseLookupUpdateState:
    """Persistent reciprocal maps, free list, and maintenance cursor."""

    index: list[list[int]]
    slot_to_index: list[list[int]]
    free_slots: list[list[int]]
    free_head: list[list[int]]


def _width(name: str, rows: Sequence[Sequence[int]]) -> int:
    if not rows:
        raise ValueError(f"{name} must contain at least one row")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be non-empty and rectangular")
    return width


def _validate_state(
    state: DSASparseLookupUpdateState,
) -> tuple[int, int, int, int]:
    pool_capacity = len(state.index)
    if (
        len(state.slot_to_index) != pool_capacity
        or len(state.free_slots) != pool_capacity
        or len(state.free_head) != pool_capacity
    ):
        raise ValueError("all state tensors must have the same row count")
    index_capacity = _width("index", state.index)
    slot_count = _width("slot_to_index", state.slot_to_index)
    free_count = _width("free_slots", state.free_slots)
    head_stride = _width("free_head", state.free_head)
    if head_stride < 2:
        raise ValueError("free_head needs head and cursor cells")
    if free_count > slot_count:
        raise ValueError("free list cannot exceed slot count")

    for row in range(pool_capacity):
        if state.free_head[row][0] != 0:
            raise ValueError(
                "fused lookup/update requires free_head[row][0] == 0 "
                "at call entry"
            )
        cursor = state.free_head[row][1]
        if not 0 <= cursor < slot_count:
            raise ValueError("maintenance cursor is outside slot range")
        if len(set(state.free_slots[row])) != free_count:
            raise ValueError("free_slots must not contain duplicates")
        for slot in state.free_slots[row]:
            if not 0 <= slot < slot_count:
                raise ValueError("free_slots contains an invalid slot")
            if state.slot_to_index[row][slot] != INVALID_INDEX:
                raise ValueError("free_slots must name unoccupied slots")
        for token, slot in enumerate(state.index[row]):
            if slot == INVALID_INDEX:
                continue
            if not 0 <= slot < slot_count:
                raise ValueError("index contains an invalid slot")
            if state.slot_to_index[row][slot] != token:
                raise ValueError("index and slot_to_index disagree")
        for slot, token in enumerate(state.slot_to_index[row]):
            if token == INVALID_INDEX:
                continue
            if not 0 <= token < index_capacity:
                raise ValueError("slot_to_index contains an invalid token")
            if state.index[row][token] != slot:
                raise ValueError("slot_to_index and index disagree")
    return pool_capacity, index_capacity, slot_count, free_count


def dsa_sparse_lookup_update_reference(
    state: DSASparseLookupUpdateState,
    *,
    req_pool_entries: Sequence[int],
    query_index: Sequence[Sequence[int]],
    lookup_mask: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Run lookup, allocate misses, and maintain the free list.

    Active valid query positions must be unique within each request. Every
    returned slot is protected from this call's maintenance phase.
    """

    pool_capacity, index_capacity, slot_count, free_count = (
        _validate_state(state)
    )
    req_num = len(req_pool_entries)
    if req_num == 0:
        raise ValueError("req_pool_entries must not be empty")
    query_width = _width("query_index", query_index)
    if len(query_index) != req_num:
        raise ValueError("query_index row count must equal req_num")
    if (
        len(lookup_mask) != req_num
        or _width("lookup_mask", lookup_mask) != query_width
    ):
        raise ValueError("lookup_mask shape must equal query_index")
    if len(set(req_pool_entries)) != req_num:
        raise ValueError("req_pool_entries must be unique in one call")
    for row in req_pool_entries:
        if not 0 <= row < pool_capacity:
            raise ValueError("req_pool_entries contains an invalid row")

    slot_out = [
        [INVALID_INDEX] * query_width for _ in range(req_num)
    ]
    miss_out = [[0] * query_width for _ in range(req_num)]

    for req_id, pool_row in enumerate(req_pool_entries):
        row_index = state.index[pool_row]
        row_slot_to_index = state.slot_to_index[pool_row]
        row_free_slots = state.free_slots[pool_row]
        cursor = state.free_head[pool_row][1]

        misses: list[tuple[int, int]] = []
        seen_tokens: set[int] = set()
        for entry, token in enumerate(query_index[req_id]):
            if lookup_mask[req_id][entry] == 0:
                continue
            if not 0 <= token < index_capacity:
                continue
            if token in seen_tokens:
                raise ValueError(
                    "active query positions must be unique per request"
                )
            seen_tokens.add(token)
            slot = row_index[token]
            if slot != INVALID_INDEX:
                slot_out[req_id][entry] = slot
                continue
            misses.append((entry, token))

        if len(misses) > free_count:
            raise RuntimeError("miss count exceeds the free-list capacity")

        for miss_rank, (entry, token) in enumerate(misses):
            slot = row_free_slots[miss_rank]
            if row_slot_to_index[slot] != INVALID_INDEX:
                raise ValueError("free list points to an occupied slot")
            row_index[token] = slot
            row_slot_to_index[slot] = token
            slot_out[req_id][entry] = slot
            miss_out[req_id][entry] = 1

        miss_count = len(misses)
        state.free_head[pool_row][0] = miss_count
        if miss_count == 0:
            state.free_head[pool_row][0] = 0
            continue

        protected = {
            slot
            for slot in slot_out[req_id]
            if slot != INVALID_INDEX
        }
        victims: list[int] = []
        for position in range(slot_count):
            slot = (cursor + position) % slot_count
            if (
                slot not in protected
                and row_slot_to_index[slot] != INVALID_INDEX
            ):
                victims.append(slot)
                if len(victims) == miss_count:
                    break
        if len(victims) != miss_count:
            raise RuntimeError(
                "not enough occupied non-protected slots to maintain "
                "the resident count"
            )

        for rank, slot in enumerate(victims):
            old_token = row_slot_to_index[slot]
            row_slot_to_index[slot] = INVALID_INDEX
            if row_index[old_token] == slot:
                row_index[old_token] = INVALID_INDEX
            row_free_slots[miss_count - 1 - rank] = slot

        state.free_head[pool_row][1] = (
            victims[-1] + 1
        ) % slot_count
        state.free_head[pool_row][0] = 0

    _validate_state(state)
    return slot_out, miss_out


__all__ = [
    "DSASparseLookupUpdateState",
    "INVALID_INDEX",
    "dsa_sparse_lookup_update_reference",
]
