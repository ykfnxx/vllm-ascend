# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass

INVALID_INDEX = -1


@dataclass
class ReferenceState:
    index: list[list[int]]
    slot_to_index: list[list[int]]
    free_slots: list[list[int]]
    free_head: list[list[int]]


def lookup_update_batch_reference(
    state: ReferenceState,
    request_rows: list[int],
    query_start_loc: list[int],
    query_positions: list[int],
    semantic_topk: list[list[list[int]]],
    *,
    block_size: int,
    tail_base: int,
    fallback_slot: int,
    staging_base: int,
    decode_mode: int,
) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    query_width = len(semantic_topk[0][0])
    index_capacity = len(state.index[0])
    slot_count = len(state.slot_to_index[0])
    free_slot_count = len(state.free_slots[0])
    resident_slots = slot_count - free_slot_count
    replaceable_base = (
        (resident_slots + block_size - 1) // block_size
    ) * block_size
    mapped = [[[-1] * query_width] for _ in semantic_topk]
    misses = [[[0] * query_width] for _ in semantic_topk]

    def map_slot(slot: int) -> int:
        if slot < resident_slots:
            return slot
        return slot - resident_slots + replaceable_base

    for request_index, state_row in enumerate(request_rows):
        query_begin = query_start_loc[request_index]
        query_end = query_start_loc[request_index + 1]
        if state_row < 0:
            for query_index in range(query_begin, query_end):
                mapped[query_index][0] = semantic_topk[query_index][0].copy()
            continue

        verify_start = query_positions[query_begin]
        tail_start = verify_start // block_size * block_size
        protected_slots: set[int] = set()
        cursor = state.free_head[state_row][1]
        for query_index in range(query_begin, query_end):
            current_position = query_positions[query_index]
            history_misses: list[tuple[int, int]] = []
            for topk_index, token_position in enumerate(
                semantic_topk[query_index][0]
            ):
                if token_position < 0 or token_position >= index_capacity:
                    continue
                if token_position < tail_start:
                    mapped[query_index][0][topk_index] = fallback_slot
                    slot = state.index[state_row][token_position]
                    if slot == INVALID_INDEX:
                        history_misses.append(
                            (topk_index, token_position)
                        )
                    else:
                        mapped[query_index][0][topk_index] = map_slot(slot)
                        protected_slots.add(slot)
                elif decode_mode == 0:
                    if token_position <= current_position:
                        mapped[query_index][0][topk_index] = (
                            tail_base + token_position - tail_start
                        )
                elif token_position < verify_start:
                    mapped[query_index][0][topk_index] = (
                        tail_base + token_position - tail_start
                    )
                elif token_position <= current_position:
                    mapped[query_index][0][topk_index] = (
                        staging_base + token_position - verify_start
                    )

            victims = [
                slot
                for offset in range(slot_count)
                if (slot := (cursor + offset) % slot_count)
                not in protected_slots
                and state.slot_to_index[state_row][slot] != INVALID_INDEX
            ]
            transaction_count = min(
                len(history_misses),
                free_slot_count,
                len(victims),
            )
            for miss_rank, (topk_index, token_position) in enumerate(
                history_misses[:transaction_count]
            ):
                slot = state.free_slots[state_row][miss_rank]
                state.index[state_row][token_position] = slot
                state.slot_to_index[state_row][slot] = token_position
                mapped[query_index][0][topk_index] = map_slot(slot)
                misses[query_index][0][topk_index] = 1
                protected_slots.add(slot)

            reclaimed_slots = victims[:transaction_count]
            for victim_rank, slot in enumerate(reclaimed_slots):
                old_token = state.slot_to_index[state_row][slot]
                state.index[state_row][old_token] = INVALID_INDEX
                state.slot_to_index[state_row][slot] = INVALID_INDEX
                state.free_slots[state_row][
                    transaction_count - 1 - victim_rank
                ] = slot
            if reclaimed_slots:
                cursor = (reclaimed_slots[-1] + 1) % slot_count
                state.free_head[state_row][1] = cursor
            state.free_head[state_row][0] = 0

    return mapped, misses
