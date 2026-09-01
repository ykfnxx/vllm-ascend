# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from .reference import (
    INVALID_INDEX,
    lookup_update_cascade_reference,
)
from .test_lookup_update_reference import make_state


def run_reference(
    state,
    semantic_topk,
    *,
    positions,
    query_start_loc,
    request_rows=None,
    decode_mode=0,
):
    return lookup_update_cascade_reference(
        state,
        [0] if request_rows is None else request_rows,
        query_start_loc,
        positions,
        semantic_topk,
        block_size=4,
        tail_base=8,
        fallback_slot=12,
        staging_base=13,
        decode_mode=decode_mode,
    )


def test_normal_history_hit_miss_tail_invalid_and_future() -> None:
    state = make_state()

    mapped, misses = run_reference(
        state,
        [[[1, 6, 8, -1, 9]]],
        positions=[8],
        query_start_loc=[0, 1],
    )

    assert mapped == [[[1, 4, 8, -1, -1]]]
    assert misses == [[[0, 1, 0, 0, 0]]]
    assert state.index[0][6] == 4
    assert state.slot_to_index[0][4] == 6


def test_mtp_maps_tail_and_staging_and_protects_across_queries() -> None:
    state = make_state()

    mapped, misses = run_reference(
        state,
        [
            [[1, 6, 8, -1]],
            [[6, 7, 8, 9]],
        ],
        positions=[8, 9],
        query_start_loc=[0, 2],
        decode_mode=1,
    )

    assert mapped == [
        [[1, 4, 13, -1]],
        [[4, 0, 13, 14]],
    ]
    assert misses == [
        [[0, 1, 0, 0]],
        [[0, 1, 0, 0]],
    ]
    assert state.index[0][6] == 4
    assert state.index[0][7] == 0
    assert state.index[0][1] == 1


def test_resource_shortage_uses_final_fallback_without_false_miss() -> None:
    state = make_state()

    mapped, misses = run_reference(
        state,
        [[[0, 1, 2, 6, 7]]],
        positions=[8],
        query_start_loc=[0, 1],
    )

    assert mapped == [[[0, 1, 2, 4, 12]]]
    assert misses == [[[0, 0, 0, 1, 0]]]
    assert state.index[0][6] == 4
    assert state.index[0][7] == INVALID_INDEX


def test_non_decode_request_is_passed_through() -> None:
    state = make_state()
    semantic = [[[3, -1, 15]]]

    mapped, misses = run_reference(
        state,
        semantic,
        positions=[3],
        query_start_loc=[0, 1],
        request_rows=[-1],
    )

    assert mapped == semantic
    assert misses == [[[0, 0, 0]]]
