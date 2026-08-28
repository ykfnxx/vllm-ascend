# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import copy

import pytest

from tests.ut.ops.dsa_sparse_lookup_update_reference import (
    DSASparseLookupUpdateState,
    INVALID_INDEX,
    dsa_sparse_lookup_update_reference,
)


def make_state(
    *,
    rows: int = 1,
    index_capacity: int = 16,
    resident_count: int = 4,
    free_count: int = 2,
) -> DSASparseLookupUpdateState:
    slot_count = resident_count + free_count
    index = [
        [INVALID_INDEX] * index_capacity for _ in range(rows)
    ]
    slot_to_index = [
        [INVALID_INDEX] * slot_count for _ in range(rows)
    ]
    for row in range(rows):
        for token in range(resident_count):
            index[row][token] = token
            slot_to_index[row][token] = token
    return DSASparseLookupUpdateState(
        index=index,
        slot_to_index=slot_to_index,
        free_slots=[
            list(range(resident_count, slot_count))
            for _ in range(rows)
        ],
        free_head=[[0, 0, 0, 0] for _ in range(rows)],
    )


def run(
    state: DSASparseLookupUpdateState,
    query: list[list[int]],
    *,
    mask: list[list[int]] | None = None,
    req_pool_entries: list[int] | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    if mask is None:
        mask = [[1] * len(row) for row in query]
    if req_pool_entries is None:
        req_pool_entries = list(range(len(query)))
    return dsa_sparse_lookup_update_reference(
        state,
        req_pool_entries=req_pool_entries,
        query_index=query,
        lookup_mask=mask,
    )


def test_hit_miss_mask_and_maintain() -> None:
    state = make_state()

    slots, misses = run(
        state,
        [[1, 6, -1, 8, 2, 3]],
        mask=[[1, 1, 1, 0, 1, 1]],
    )

    assert slots == [[1, 4, -1, -1, 2, 3]]
    assert misses == [[0, 1, 0, 0, 0, 0]]
    assert state.index[0][6] == 4
    assert state.slot_to_index[0][4] == 6
    assert state.index[0][0] == INVALID_INDEX
    assert state.slot_to_index[0][0] == INVALID_INDEX
    assert state.free_slots[0][0] == 0
    assert state.free_head[0][0] == 0
    assert state.free_head[0][1] == 1


def test_cursor_advances_and_protected_hits_are_not_evicted() -> None:
    state = make_state()
    run(state, [[1, 6]])

    slots, misses = run(state, [[1, 7]])

    assert slots == [[1, 0]]
    assert misses == [[0, 1]]
    assert state.index[0][1] == 1
    assert state.index[0][2] == INVALID_INDEX
    assert state.free_slots[0][0] == 2
    assert state.free_head[0][1] == 3


def test_all_hit_does_not_mutate_state_or_cursor() -> None:
    state = make_state()
    before = copy.deepcopy(state)

    slots, misses = run(state, [[3, 1, 2]])

    assert slots == [[3, 1, 2]]
    assert misses == [[0, 0, 0]]
    assert state == before


def test_reordered_pool_rows_address_stable_state_rows() -> None:
    state = make_state(rows=2)
    state.index[0][1] = INVALID_INDEX
    state.slot_to_index[0][1] = INVALID_INDEX
    state.index[0][9] = 1
    state.slot_to_index[0][1] = 9

    slots, misses = run(
        state,
        [[2], [9]],
        req_pool_entries=[1, 0],
    )

    assert slots == [[2], [1]]
    assert misses == [[0], [0]]


def test_multiple_misses_allocate_and_refill_in_input_order() -> None:
    state = make_state(resident_count=4, free_count=3)

    slots, misses = run(state, [[8, 7, 6]])

    assert slots == [[4, 5, 6]]
    assert misses == [[1, 1, 1]]
    assert state.free_slots[0] == [2, 1, 0]
    assert state.free_head[0][:2] == [0, 3]


def test_duplicate_pool_entries_are_rejected() -> None:
    state = make_state(rows=2)

    with pytest.raises(ValueError, match="must be unique"):
        run(
            state,
            [[1], [2]],
            req_pool_entries=[0, 0],
        )


def test_duplicate_active_query_positions_are_rejected() -> None:
    state = make_state()

    with pytest.raises(ValueError, match="active query positions"):
        run(state, [[1, 6, 6]])


def test_nonzero_entry_head_is_rejected() -> None:
    state = make_state()
    state.free_head[0][0] = 1

    with pytest.raises(ValueError, match=r"free_head\[row\]\[0\]"):
        run(state, [[1]])
