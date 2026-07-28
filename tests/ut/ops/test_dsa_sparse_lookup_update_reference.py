# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Standard-library tests for the DSA sparse lookup/update CPU oracle."""

from __future__ import annotations

import copy
import unittest

from tests.ut.ops.dsa_sparse_lookup_update_reference import (
    DSASparseLookupUpdateState,
    INVALID_INDEX,
    dsa_sparse_lookup_update_reference,
)


def make_state(
    *,
    num_requests: int = 1,
    max_model_len: int = 16,
    slot_count: int = 4,
) -> DSASparseLookupUpdateState:
    return DSASparseLookupUpdateState(
        token_to_hot=[[INVALID_INDEX] * max_model_len
                      for _ in range(num_requests)],
        hot_to_token=[[INVALID_INDEX] * slot_count
                      for _ in range(num_requests)],
        lru_slots=[list(range(slot_count)) for _ in range(num_requests)],
    )


def install(state: DSASparseLookupUpdateState, *, request_index: int, slot: int,
            token: int) -> None:
    if state.hot_to_token[request_index][slot] != INVALID_INDEX:
        raise AssertionError("test helper cannot overwrite an occupied slot")
    if state.token_to_hot[request_index][token] != INVALID_INDEX:
        raise AssertionError("test helper cannot install a token twice")
    state.hot_to_token[request_index][slot] = token
    state.token_to_hot[request_index][token] = slot


def run_one_row(
    state: DSASparseLookupUpdateState,
    topk_positions: list[list[int]],
    *,
    query_positions: list[int] | None = None,
    query_to_lane: list[int] | None = None,
    query_valid_mask: list[bool] | None = None,
    valid_topk_counts: list[int] | None = None,
    seq_len: int = 12,
) -> tuple[list[list[int]], list[list[bool]]]:
    num_queries = len(topk_positions)
    width = len(topk_positions[0]) if topk_positions else 0
    if query_positions is None:
        query_positions = list(
            range(seq_len - num_queries, seq_len))
    if query_to_lane is None:
        query_to_lane = list(range(num_queries))
    if query_valid_mask is None:
        query_valid_mask = [True] * num_queries
    if valid_topk_counts is None:
        valid_topk_counts = [width] * num_queries
    return dsa_sparse_lookup_update_reference(
        state,
        query_positions=query_positions,
        query_to_req_idx=[0] * num_queries,
        query_to_lane=query_to_lane,
        query_valid_mask=query_valid_mask,
        valid_topk_counts=valid_topk_counts,
        seq_lens=[seq_len],
        topk_positions=topk_positions,
    )


class TestDSASparseLookupUpdateReference(unittest.TestCase):

    def test_all_none_and_mixed_residency(self) -> None:
        cases = []

        all_resident = make_state(slot_count=4)
        install(all_resident, request_index=0, slot=0, token=2)
        install(all_resident, request_index=0, slot=2, token=5)
        all_resident.lru_slots[0] = [1, 3, 0, 2]
        cases.append((
            "all resident",
            all_resident,
            [[2, 5]],
            [[0, 2]],
            [[False, False]],
            [1, 3, 0, 2],
        ))

        none_resident = make_state(slot_count=4)
        none_resident.lru_slots[0] = [2, 0, 3, 1]
        cases.append((
            "none resident",
            none_resident,
            [[2, 5]],
            [[2, 0]],
            [[True, True]],
            [3, 1, 2, 0],
        ))

        mixed = make_state(slot_count=4)
        install(mixed, request_index=0, slot=3, token=2)
        mixed.lru_slots[0] = [1, 2, 3, 0]
        cases.append((
            "mixed",
            mixed,
            [[2, 5]],
            [[3, 1]],
            [[False, True]],
            [2, 0, 1, 3],
        ))

        for (name, state, topk, expected_slots, expected_misses,
             expected_lru) in cases:
            with self.subTest(name=name):
                slots, misses = run_one_row(state, topk)
                self.assertEqual(slots, expected_slots)
                self.assertEqual(misses, expected_misses)
                self.assertEqual(state.lru_slots[0], expected_lru)

    def test_duplicate_nonresident_uses_first_flat_occurrence(self) -> None:
        state = make_state(slot_count=3)

        slots, misses = run_one_row(
            state,
            [[7, 7], [7, 8]],
            query_positions=[10, 11],
            seq_len=12,
        )

        self.assertEqual(slots, [[0, 0], [0, 1]])
        self.assertEqual(misses, [[True, False], [False, True]])
        self.assertEqual(state.token_to_hot[0][7], 0)
        self.assertEqual(state.token_to_hot[0][8], 1)
        self.assertEqual(state.hot_to_token[0], [7, 8, INVALID_INDEX])
        self.assertEqual(state.lru_slots[0], [2, 0, 1])

    def test_preserves_asu_single_query_lookup_lru_contract(self) -> None:
        """Lock the original ASU SIMT lookup/allocate/evict behavior."""
        state = make_state(max_model_len=8, slot_count=5)
        for slot, token in enumerate([0, 1, 2, 3, 4]):
            install(state, request_index=0, slot=slot, token=token)
        state.lru_slots[0] = [3, 0, 4, 1, 2]

        slots, misses = run_one_row(
            state,
            [[2, 5, 5, INVALID_INDEX, 1]],
            query_positions=[7],
            seq_len=8,
        )

        self.assertEqual(slots, [[2, 3, 3, INVALID_INDEX, 1]])
        self.assertEqual(
            misses,
            [[False, True, False, False, False]],
        )
        self.assertEqual(state.token_to_hot[0][3], INVALID_INDEX)
        self.assertEqual(state.token_to_hot[0][5], 3)
        self.assertEqual(state.hot_to_token[0], [0, 1, 2, 5, 4])
        self.assertEqual(state.lru_slots[0], [0, 4, 3, 1, 2])

    def test_padding_validity_and_inactive_request_stay_invalid(self) -> None:
        state = make_state(num_requests=2, slot_count=3)
        install(state, request_index=0, slot=2, token=1)
        install(state, request_index=1, slot=1, token=3)
        state.lru_slots[0] = [0, 1, 2]
        state.lru_slots[1] = [2, 0, 1]
        inactive_before = copy.deepcopy((
            state.token_to_hot[1],
            state.hot_to_token[1],
            state.lru_slots[1],
        ))

        slots, misses = dsa_sparse_lookup_update_reference(
            state,
            query_positions=[5, 5, 4],
            query_to_req_idx=[0, 0, 1],
            query_to_lane=[0, 1, 0],
            query_valid_mask=[True, False, False],
            valid_topk_counts=[2, 4, 4],
            seq_lens=[6, 5],
            topk_positions=[
                [1, 6, -1, 4],
                [1, 2, 3, 4],
                [0, 1, 2, 3],
            ],
        )

        self.assertEqual(slots, [
            [2, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX],
            [INVALID_INDEX] * 4,
            [INVALID_INDEX] * 4,
        ])
        self.assertEqual(misses, [[False] * 4, [False] * 4, [False] * 4])
        self.assertEqual(
            (
                state.token_to_hot[1],
                state.hot_to_token[1],
                state.lru_slots[1],
            ),
            inactive_before,
        )

    def test_duplicate_request_lane_mapping_is_rejected(self) -> None:
        state = make_state(slot_count=4)

        with self.assertRaisesRegex(
            ValueError,
            r"\(request index, query lane\)",
        ):
            dsa_sparse_lookup_update_reference(
                state,
                query_positions=[4, 5],
                query_to_req_idx=[0, 0],
                query_to_lane=[0, 0],
                query_valid_mask=[True, True],
                valid_topk_counts=[1, 1],
                seq_lens=[6],
                topk_positions=[[1], [2]],
            )

    def test_free_slot_precedes_lru_eviction(self) -> None:
        state = make_state(max_model_len=8, slot_count=3)
        install(state, request_index=0, slot=0, token=0)
        install(state, request_index=0, slot=1, token=1)
        state.lru_slots[0] = [0, 1, 2]

        slots, misses = run_one_row(
            state,
            [[3]],
            query_positions=[4],
            seq_len=5,
        )

        self.assertEqual(slots, [[2]])
        self.assertEqual(misses, [[True]])
        self.assertEqual(state.hot_to_token[0], [0, 1, 3])
        self.assertEqual(state.lru_slots[0], [0, 1, 2])

    def test_eviction_clears_the_old_forward_mapping(self) -> None:
        state = make_state(max_model_len=8, slot_count=2)
        install(state, request_index=0, slot=0, token=1)
        install(state, request_index=0, slot=1, token=2)
        state.lru_slots[0] = [0, 1]

        slots, misses = run_one_row(
            state,
            [[3]],
            query_positions=[4],
            seq_len=5,
        )

        self.assertEqual(slots, [[0]])
        self.assertEqual(misses, [[True]])
        self.assertEqual(state.token_to_hot[0][1], INVALID_INDEX)
        self.assertEqual(state.token_to_hot[0][2], 1)
        self.assertEqual(state.token_to_hot[0][3], 0)
        self.assertEqual(state.hot_to_token[0], [3, 2])
        self.assertEqual(state.lru_slots[0], [1, 0])

    def test_reordered_query_view_uses_stable_request_index_state(self) -> None:
        state = make_state(
            num_requests=2,
            max_model_len=10,
            slot_count=2,
        )
        install(state, request_index=0, slot=1, token=2)
        install(state, request_index=1, slot=0, token=3)

        slots, misses = dsa_sparse_lookup_update_reference(
            state,
            query_positions=[7, 8],
            query_to_req_idx=[1, 0],
            query_to_lane=[0, 0],
            query_valid_mask=[True, True],
            valid_topk_counts=[1, 1],
            seq_lens=[9, 9],
            topk_positions=[[3], [2]],
        )

        self.assertEqual(slots, [[0], [1]])
        self.assertEqual(misses, [[False], [False]])
        self.assertEqual(state.token_to_hot[0][2], 1)
        self.assertEqual(state.hot_to_token[0][1], 2)
        self.assertEqual(state.token_to_hot[1][3], 0)
        self.assertEqual(state.hot_to_token[1][0], 3)

    def test_newest_uses_reserved_slots_and_invalidates_stale_mapping(
            self) -> None:
        state = make_state(max_model_len=12, slot_count=3)
        install(state, request_index=0, slot=0, token=2)
        install(state, request_index=0, slot=1, token=9)
        state.lru_slots[0] = [1, 0, 2]

        slots, misses = run_one_row(
            state,
            [[9, 10, 7], [10, 9, 7]],
            query_positions=[9, 10],
            query_to_lane=[0, 1],
            seq_len=11,
        )

        self.assertEqual(slots, [[3, 4, 1], [4, 3, 1]])
        self.assertEqual(misses, [[False, False, True],
                                  [False, False, False]])
        self.assertEqual(state.token_to_hot[0][9], INVALID_INDEX)
        self.assertEqual(state.token_to_hot[0][10], INVALID_INDEX)
        self.assertEqual(state.token_to_hot[0][7], 1)
        self.assertEqual(state.hot_to_token[0], [2, 7, INVALID_INDEX])
        self.assertEqual(state.lru_slots[0], [0, 2, 1])

    def test_newest_released_slot_is_free_first_even_when_mru(self) -> None:
        state = make_state(max_model_len=12, slot_count=3)
        install(state, request_index=0, slot=0, token=1)
        install(state, request_index=0, slot=1, token=2)
        install(state, request_index=0, slot=2, token=9)
        state.lru_slots[0] = [0, 1, 2]

        slots, misses = run_one_row(
            state,
            [[4]],
            query_positions=[9],
            seq_len=10,
        )

        self.assertEqual(slots, [[2]])
        self.assertEqual(misses, [[True]])
        self.assertEqual(state.hot_to_token[0], [1, 2, 4])
        self.assertEqual(state.token_to_hot[0][1], 0)
        self.assertEqual(state.token_to_hot[0][2], 1)
        self.assertEqual(state.token_to_hot[0][9], INVALID_INDEX)
        self.assertEqual(state.lru_slots[0], [0, 1, 2])

    def test_mtp_union_protects_hits_and_updates_exact_lru_order(self) -> None:
        state = make_state(max_model_len=12, slot_count=5)
        for slot, token in enumerate([1, 2, 3, 4, 5]):
            install(state, request_index=0, slot=slot, token=token)
        state.lru_slots[0] = [4, 0, 1, 3, 2]

        slots, misses = run_one_row(
            state,
            [[2, 6, 7], [4, 6, 7]],
            query_positions=[8, 9],
            query_to_lane=[0, 1],
            seq_len=10,
        )

        self.assertEqual(slots, [[1, 4, 0], [3, 4, 0]])
        self.assertEqual(misses, [[False, True, True],
                                  [False, False, False]])
        self.assertEqual(state.token_to_hot[0][5], INVALID_INDEX)
        self.assertEqual(state.token_to_hot[0][1], INVALID_INDEX)
        self.assertEqual(state.token_to_hot[0][3], 2)
        self.assertEqual(state.token_to_hot[0][6], 4)
        self.assertEqual(state.token_to_hot[0][7], 0)
        self.assertEqual(state.hot_to_token[0], [7, 2, 3, 4, 6])
        # stale + newly allocated + hits, each preserving its specified order
        self.assertEqual(state.lru_slots[0], [2, 4, 0, 1, 3])


if __name__ == "__main__":
    unittest.main()
