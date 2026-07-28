# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Deterministic CPU reference for the fused DSA sparse lookup/update ABI.

Persistent metadata is indexed directly by stable request index. This oracle
models only metadata semantics; payload I/O remains outside the operator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


INVALID_INDEX = -1


@dataclass
class DSASparseLookupUpdateState:
    """Persistent metadata indexed by stable request index.

    ``token_to_hot`` and ``hot_to_token`` are reciprocal maps.  ``lru_slots``
    is ordered from least-recently-used to most-recently-used.
    """

    token_to_hot: list[list[int]]
    hot_to_token: list[list[int]]
    lru_slots: list[list[int]]


def _rectangular_width(name: str, values: Sequence[Sequence[int]]) -> int:
    if not values:
        return 0
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError(f"{name} must be rectangular")
    return width


def _validate_state(state: DSASparseLookupUpdateState) -> tuple[int, int, int]:
    num_requests = len(state.token_to_hot)
    if len(state.hot_to_token) != num_requests:
        raise ValueError("hot_to_token request count does not match token_to_hot")
    if len(state.lru_slots) != num_requests:
        raise ValueError("lru_slots request count does not match token_to_hot")

    max_model_len = _rectangular_width("token_to_hot",
                                       state.token_to_hot)
    slot_count = _rectangular_width("hot_to_token", state.hot_to_token)
    if _rectangular_width("lru_slots", state.lru_slots) != slot_count:
        raise ValueError("lru_slots width does not match hot_to_token")

    expected_slots = list(range(slot_count))
    for request_index in range(num_requests):
        if sorted(state.lru_slots[request_index]) != expected_slots:
            raise ValueError(
                f"lru_slots[{request_index}] must be a permutation of all slots")

        for token, slot in enumerate(state.token_to_hot[request_index]):
            if slot == INVALID_INDEX:
                continue
            if not 0 <= slot < slot_count:
                raise ValueError(
                    f"token_to_hot[{request_index}][{token}] has invalid slot {slot}")
            if state.hot_to_token[request_index][slot] != token:
                raise ValueError("token_to_hot and hot_to_token disagree")

        for slot, token in enumerate(state.hot_to_token[request_index]):
            if token == INVALID_INDEX:
                continue
            if not 0 <= token < max_model_len:
                raise ValueError(
                    f"hot_to_token[{request_index}][{slot}] has invalid token {token}")
            if state.token_to_hot[request_index][token] != slot:
                raise ValueError("hot_to_token and token_to_hot disagree")

    return num_requests, max_model_len, slot_count


def dsa_sparse_lookup_update_reference(
    state: DSASparseLookupUpdateState,
    *,
    query_positions: Sequence[int],
    query_to_req_idx: Sequence[int],
    query_to_lane: Sequence[int],
    query_valid_mask: Sequence[bool],
    valid_topk_counts: Sequence[int],
    seq_lens: Sequence[int],
    topk_positions: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[list[bool]]]:
    """Apply one fused lookup/update step and mutate ``state`` in place.

    The flattened ``(query, top-k entry)`` order is the canonical order for
    duplicate misses and allocations.  Entries that point at a valid current
    query position use the reserved index ``slot_count + query_lane`` and are
    never installed in the evictable maps.
    """

    num_requests, max_model_len, slot_count = _validate_state(state)
    if len(seq_lens) != num_requests:
        raise ValueError("seq_lens must contain one value per request index")

    num_queries = len(topk_positions)
    query_vectors = (
        query_positions,
        query_to_req_idx,
        query_to_lane,
        query_valid_mask,
        valid_topk_counts,
    )
    if any(len(values) != num_queries for values in query_vectors):
        raise ValueError("query metadata lengths must match topk_positions")

    topk_width = _rectangular_width("topk_positions", topk_positions)
    for query, count in enumerate(valid_topk_counts):
        if not 0 <= count <= topk_width:
            raise ValueError(
                f"valid_topk_counts[{query}] is outside [0, {topk_width}]")

    for request_index, seq_len in enumerate(seq_lens):
        if not 0 <= seq_len <= max_model_len:
            raise ValueError(
                f"seq_lens[{request_index}] is outside [0, {max_model_len}]")

    request_lane_owners: set[tuple[int, int]] = set()
    for query, request_index in enumerate(query_to_req_idx):
        if not 0 <= request_index < num_requests:
            raise ValueError(
                f"query_to_req_idx[{query}] has invalid request index "
                f"{request_index}")
        lane = query_to_lane[query]
        if lane < 0:
            raise ValueError(f"query_to_lane[{query}] must be non-negative")
        request_lane = (request_index, lane)
        if request_lane in request_lane_owners:
            raise ValueError(
                "each (request index, query lane) may own at most one query"
            )
        request_lane_owners.add(request_lane)

    slot_out = [[INVALID_INDEX] * topk_width
                for _ in range(num_queries)]
    miss_mask = [[False] * topk_width for _ in range(num_queries)]

    queries_by_request: list[list[int]] = [
        [] for _ in range(num_requests)
    ]
    for query, request_index in enumerate(query_to_req_idx):
        queries_by_request[request_index].append(query)

    for request_index in range(num_requests):
        request_queries = queries_by_request[request_index]
        if not any(query_valid_mask[query] for query in request_queries):
            continue

        seq_len = seq_lens[request_index]

        # The first valid flat query occurrence defines the reserved lane if
        # malformed/padded metadata repeats a current position.
        newest_lane_by_token: dict[int, int] = {}
        for query in request_queries:
            token = query_positions[query]
            if (query_valid_mask[query] and 0 <= token < seq_len
                    and token < max_model_len):
                newest_lane_by_token.setdefault(token,
                                                query_to_lane[query])

        # A position may have been evictable in the preceding step and become
        # a current query position now.  Remove that stale long-term mapping
        # before classifying this step's TopK entries.
        for token in newest_lane_by_token:
            old_slot = state.token_to_hot[request_index][token]
            if old_slot == INVALID_INDEX:
                continue
            state.token_to_hot[request_index][token] = INVALID_INDEX
            if state.hot_to_token[request_index][old_slot] == token:
                state.hot_to_token[request_index][old_slot] = INVALID_INDEX

        old_lru = list(state.lru_slots[request_index])
        hit_slots: set[int] = set()
        missing_occurrences: dict[int, list[tuple[int, int]]] = {}
        missing_order: list[int] = []

        for query in request_queries:
            if not query_valid_mask[query]:
                continue
            valid_count = valid_topk_counts[query]
            for topk_idx in range(valid_count):
                token = topk_positions[query][topk_idx]
                if not 0 <= token < seq_len or token >= max_model_len:
                    continue

                newest_lane = newest_lane_by_token.get(token)
                if newest_lane is not None:
                    slot_out[query][topk_idx] = slot_count + newest_lane
                    continue

                resident_slot = state.token_to_hot[request_index][token]
                if resident_slot != INVALID_INDEX:
                    slot_out[query][topk_idx] = resident_slot
                    hit_slots.add(resident_slot)
                    continue

                if token not in missing_occurrences:
                    missing_occurrences[token] = []
                    missing_order.append(token)
                missing_occurrences[token].append((query, topk_idx))

        # All MTP queries belonging to the request are classified before choosing
        # victims, so a hit in any query protects the slot from this step.
        evictable_lru = [
            slot for slot in old_lru if slot not in hit_slots
        ]
        free_slots = [
            slot for slot in evictable_lru
            if state.hot_to_token[request_index][slot] == INVALID_INDEX
        ]
        occupied_slots = [
            slot for slot in evictable_lru
            if state.hot_to_token[request_index][slot] != INVALID_INDEX
        ]
        allocation_candidates = free_slots + occupied_slots
        if len(missing_order) > len(allocation_candidates):
            raise RuntimeError(
                "not enough evictable slots for this request's canonical misses")

        allocated_slots: list[int] = []
        allocated_by_token: dict[int, int] = {}
        for token, slot in zip(missing_order, allocation_candidates):
            evicted_token = state.hot_to_token[request_index][slot]
            if evicted_token != INVALID_INDEX:
                state.token_to_hot[request_index][evicted_token] = INVALID_INDEX

            state.hot_to_token[request_index][slot] = token
            state.token_to_hot[request_index][token] = slot
            allocated_slots.append(slot)
            allocated_by_token[token] = slot

        for token in missing_order:
            slot = allocated_by_token[token]
            occurrences = missing_occurrences[token]
            for query, topk_idx in occurrences:
                slot_out[query][topk_idx] = slot
            canonical_query, canonical_topk_idx = occurrences[0]
            miss_mask[canonical_query][canonical_topk_idx] = True

        allocated_set = set(allocated_slots)
        untouched_stale = [
            slot for slot in old_lru
            if slot not in hit_slots and slot not in allocated_set
        ]
        hits_in_lru_order = [
            slot for slot in old_lru if slot in hit_slots
        ]
        state.lru_slots[request_index][:] = (
            untouched_stale + allocated_slots + hits_in_lru_order)

    _validate_state(state)
    return slot_out, miss_mask


__all__ = [
    "DSASparseLookupUpdateState",
    "INVALID_INDEX",
    "dsa_sparse_lookup_update_reference",
]
