# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import copy

from .reference import INVALID_INDEX, lookup_update_batch_reference, lookup_update_reference
from .test_lookup_update_reference import make_state


def test_protected_slots_accumulate_across_queries() -> None:
    state = make_state()

    slots, misses = lookup_update_batch_reference(
        state,
        [0],
        [0, 2],
        [[1, 6], [6, 7]],
        [[1, 1], [1, 1]],
    )

    assert slots == [[1, 4], [4, 0]]
    assert misses == [[0, 1], [0, 1]]
    assert state.index[0][6] == 4
    assert state.index[0][7] == 0
    assert state.index[0][1] == 1


def test_resource_shortage_returns_fallback_without_installing_token() -> None:
    state = make_state()

    slots, misses = lookup_update_batch_reference(
        state,
        [0],
        [0, 2],
        [[0, 1, 2], [3, 7, -1]],
        [[1, 1, 1], [1, 1, 1]],
    )

    assert slots == [[0, 1, 2], [3, 10240, -1]]
    assert misses == [[0, 0, 0], [0, 0, 0]]
    assert state.index[0][7] == INVALID_INDEX
    assert state.free_slots[0] == [4, 5]


def test_partial_transaction_does_not_install_overflow_suffix() -> None:
    state = make_state()

    slots, misses = lookup_update_batch_reference(
        state,
        [0],
        [0, 1],
        [[0, 1, 2, 6, 7]],
        [[1, 1, 1, 1, 1]],
    )

    assert slots == [[0, 1, 2, 4, 10240]]
    assert misses == [[0, 0, 0, 1, 0]]
    assert state.index[0][6] == 4
    assert state.index[0][7] == INVALID_INDEX


def test_masked_and_out_of_range_entries_are_inactive() -> None:
    state = make_state()

    slots, misses = lookup_update_batch_reference(state, [0], [0, 1], [[6, -1, 17]], [[0, 1, 1]])

    assert slots == [[-1, -1, -1]]
    assert misses == [[0, 0, 0]]
    assert state.index[0][6] == INVALID_INDEX


def test_one_query_per_request_matches_single_query_contract() -> None:
    single_state = make_state(rows=2)
    batch_state = copy.deepcopy(single_state)
    request_rows = [1, 0]
    query_indices = [[1, 6, -1], [2, 7, 3]]
    lookup_mask = [[1, 1, 1], [1, 1, 1]]

    single_output = lookup_update_reference(single_state, request_rows, query_indices, lookup_mask)
    batch_output = lookup_update_batch_reference(
        batch_state,
        request_rows,
        [0, 1, 2],
        query_indices,
        lookup_mask,
    )

    assert batch_output == single_output
    assert batch_state == single_state
