# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import copy

from tests.ut.ops.dsa_sparse_lookup_update_batch_reference import (
    dsa_sparse_lookup_update_batch_reference,
)
from tests.ut.ops.dsa_sparse_lookup_update_reference import (
    DSASparseLookupUpdateState,
    INVALID_INDEX,
    dsa_sparse_lookup_update_reference,
)


def make_state(rows: int = 1) -> DSASparseLookupUpdateState:
    index = [[INVALID_INDEX] * 16 for _ in range(rows)]
    slot_to_index = [[INVALID_INDEX] * 6 for _ in range(rows)]
    for row in range(rows):
        for token in range(4):
            index[row][token] = token
            slot_to_index[row][token] = token
    return DSASparseLookupUpdateState(
        index=index,
        slot_to_index=slot_to_index,
        free_slots=[[4, 5] for _ in range(rows)],
        free_head=[[0, 0, 0, 0] for _ in range(rows)],
    )


def run(
    state: DSASparseLookupUpdateState,
    query: list[list[int]],
    query_start_loc: list[int],
) -> tuple[list[list[int]], list[list[int]]]:
    return dsa_sparse_lookup_update_batch_reference(
        state,
        req_pool_entries=[0],
        query_start_loc=query_start_loc,
        query_index=query,
        lookup_mask=[[1] * len(row) for row in query],
        fallback_slot=10240,
    )


def test_cross_query_duplicate_reuses_the_first_miss_mapping():
    state = make_state()

    slots, misses = run(state, [[1, 6], [6, 7]], [0, 2])

    assert slots == [[1, 4], [4, 0]]
    assert misses == [[0, 1], [0, 1]]
    assert state.index[0][6] == 4
    assert state.index[0][7] == 0
    assert state.index[0][1] == 1
    assert state.free_slots[0][0] == 2
    assert state.free_head[0][:2] == [0, 3]


def test_protected_union_can_force_fail_closed_fallback():
    state = make_state()

    slots, misses = run(
        state,
        [[0, 1, 2], [3, 7, -1]],
        [0, 2],
    )

    assert slots == [[0, 1, 2], [3, 10240, -1]]
    assert misses == [[0, 0, 0], [0, 0, 0]]
    assert state.index[0][7] == INVALID_INDEX
    assert state.free_slots[0] == [4, 5]
    assert state.free_head[0][:2] == [0, 0]


def test_partial_safe_allocation_does_not_install_the_overflow_suffix():
    state = make_state()

    slots, misses = run(
        state,
        [[0, 1, 2, 6, 7]],
        [0, 1],
    )

    assert slots == [[0, 1, 2, 4, 10240]]
    assert misses == [[0, 0, 0, 1, 0]]
    assert state.index[0][6] == 4
    assert state.index[0][7] == INVALID_INDEX
    assert state.free_slots[0] == [3, 5]
    assert state.free_head[0][:2] == [0, 4]


def test_masked_and_invalid_entries_never_mutate_state():
    state = make_state()

    slots, misses = dsa_sparse_lookup_update_batch_reference(
        state,
        req_pool_entries=[0],
        query_start_loc=[0, 1],
        query_index=[[6, -1, 17]],
        lookup_mask=[[0, 1, 1]],
        fallback_slot=10240,
    )

    assert slots == [[-1, -1, -1]]
    assert misses == [[0, 0, 0]]
    assert state.index[0][6] == INVALID_INDEX


def test_variable_query_counts_and_reordered_request_rows_are_independent():
    state = make_state(rows=2)

    slots, misses = dsa_sparse_lookup_update_batch_reference(
        state,
        req_pool_entries=[1, 0],
        query_start_loc=[0, 2, 3],
        query_index=[[1, 6], [6, 7], [2, 8]],
        lookup_mask=[[1, 1], [1, 1], [1, 1]],
        fallback_slot=10240,
    )

    assert slots == [[1, 4], [4, 0], [2, 4]]
    assert misses == [[0, 1], [0, 1], [0, 1]]
    assert state.index[1][7] == 0
    assert state.index[0][8] == 4


def test_one_query_per_request_matches_the_existing_operator_contract():
    old_state = make_state(rows=2)
    batch_state = copy.deepcopy(old_state)
    req_pool_entries = [1, 0]
    query_index = [[1, 6, -1], [2, 7, 3]]
    lookup_mask = [[1, 1, 1], [1, 1, 1]]

    old_output = dsa_sparse_lookup_update_reference(
        old_state,
        req_pool_entries=req_pool_entries,
        query_index=query_index,
        lookup_mask=lookup_mask,
    )
    batch_output = dsa_sparse_lookup_update_batch_reference(
        batch_state,
        req_pool_entries=req_pool_entries,
        query_start_loc=[0, 1, 2],
        query_index=query_index,
        lookup_mask=lookup_mask,
        fallback_slot=10240,
    )

    assert batch_output == old_output
    assert batch_state == old_state
