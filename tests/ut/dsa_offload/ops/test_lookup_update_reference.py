# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import copy

from .reference import INVALID_INDEX, ReferenceState, lookup_update_reference


def make_state(rows: int = 1) -> ReferenceState:
    index = [[INVALID_INDEX] * 16 for _ in range(rows)]
    slot_to_index = [[INVALID_INDEX] * 6 for _ in range(rows)]
    for row in range(rows):
        for token_position in range(4):
            index[row][token_position] = token_position
            slot_to_index[row][token_position] = token_position
    return ReferenceState(
        index=index,
        slot_to_index=slot_to_index,
        free_slots=[[4, 5] for _ in range(rows)],
        free_head=[[0, 0, 0, 0] for _ in range(rows)],
    )


def test_hit_miss_mask_reclaim_and_bidirectional_state() -> None:
    state = make_state()

    slots, misses = lookup_update_reference(
        state,
        [0],
        [[1, 6, -1, 8, 2, 3]],
        [[1, 1, 1, 0, 1, 1]],
    )

    assert slots == [[1, 4, -1, -1, 2, 3]]
    assert misses == [[0, 1, 0, 0, 0, 0]]
    assert state.index[0][6] == 4
    assert state.slot_to_index[0][4] == 6
    assert state.index[0][0] == INVALID_INDEX
    assert state.slot_to_index[0][0] == INVALID_INDEX
    assert state.free_slots[0] == [0, 5]
    assert state.free_head[0][:2] == [0, 1]


def test_replaceable_hit_is_not_reloaded() -> None:
    state = make_state()
    lookup_update_reference(state, [0], [[6]], [[1]])

    slots, misses = lookup_update_reference(state, [0], [[6]], [[1]])

    assert slots == [[4]]
    assert misses == [[0]]


def test_all_hit_preserves_state() -> None:
    state = make_state()
    before = copy.deepcopy(state)

    assert lookup_update_reference(state, [0], [[3, 1, 2]], [[1, 1, 1]]) == (
        [[3, 1, 2]],
        [[0, 0, 0]],
    )
    assert state == before


def test_request_rows_address_stable_state_rows() -> None:
    state = make_state(rows=2)
    state.index[0][1] = INVALID_INDEX
    state.index[0][9] = 1
    state.slot_to_index[0][1] = 9

    slots, misses = lookup_update_reference(state, [1, 0], [[2], [9]], [[1], [1]])

    assert slots == [[2], [1]]
    assert misses == [[0], [0]]
