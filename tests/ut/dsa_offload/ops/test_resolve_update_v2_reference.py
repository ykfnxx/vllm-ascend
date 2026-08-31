# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import copy

from .reference import (
    FALLBACK_SLOT,
    INVALID_INDEX,
    STAGING_BASE,
    TAIL_BASE,
    resolve_update_v2_reference,
)
from .test_lookup_update_reference import make_state


def test_normal_classifies_history_tail_future_and_invalid() -> None:
    state = make_state()
    mapped, gather = resolve_update_v2_reference(
        state,
        [0],
        [0, 1],
        [6],
        [[1, 6, 7, -1, 17]],
        block_size=4,
        decode_mode=0,
    )

    assert mapped == [[1, TAIL_BASE + 2, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX]]
    assert gather == [[0, 0, 0, 0, 0]]


def test_mtp_maps_committed_tail_and_staging() -> None:
    state = make_state()
    mapped, gather = resolve_update_v2_reference(
        state,
        [0],
        [0, 2],
        [6, 7],
        [[1, 5, 6, 7], [6, 7, 8, -1]],
        block_size=4,
        decode_mode=1,
    )

    assert mapped[0] == [1, TAIL_BASE + 1, STAGING_BASE, INVALID_INDEX]
    assert mapped[1] == [STAGING_BASE, STAGING_BASE + 1, INVALID_INDEX, INVALID_INDEX]
    assert gather == [[0, 0, 0, 0], [0, 0, 0, 0]]


def test_history_miss_maps_allocated_slot_and_sets_gather() -> None:
    state = make_state()
    mapped, gather = resolve_update_v2_reference(
        state,
        [0],
        [0, 1],
        [8],
        [[6]],
        block_size=4,
        decode_mode=0,
    )

    assert mapped == [[4]]
    assert gather == [[1]]


def test_inactive_and_padding_rows_are_overwritten() -> None:
    state = make_state(rows=2)
    original = copy.deepcopy(state)
    mapped, gather = resolve_update_v2_reference(
        state,
        [-1, 1],
        [0, 1, 2],
        [8, 12, 13],
        [[1], [2], [3]],
        block_size=4,
        decode_mode=0,
    )

    assert mapped[0] == [INVALID_INDEX]
    assert mapped[2] == [INVALID_INDEX]
    assert gather[0] == [0]
    assert state.index[1] == original.index[1]


def test_resource_shortage_maps_history_to_fallback_slot() -> None:
    state = make_state()
    mapped, gather = resolve_update_v2_reference(
        state,
        [0],
        [0, 1],
        [8],
        [[0, 1, 2, 3, 7]],
        block_size=4,
        decode_mode=0,
    )

    assert mapped == [[0, 1, 2, 3, FALLBACK_SLOT]]
    assert gather == [[0, 0, 0, 0, 0]]
