# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass

INVALID_INDEX = -1
FALLBACK_SENTINEL = 10 * 1024


@dataclass
class ReferenceState:
    index: list[list[int]]
    slot_to_index: list[list[int]]
    free_slots: list[list[int]]
    free_head: list[list[int]]


def lookup_update_reference(
    state: ReferenceState,
    request_rows: list[int],
    query_indices: list[list[int]],
    lookup_mask: list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    query_width = len(query_indices[0])
    index_capacity = len(state.index[0])
    slot_count = len(state.slot_to_index[0])
    slot_out = [[INVALID_INDEX] * query_width for _ in query_indices]
    miss_out = [[0] * query_width for _ in query_indices]

    for request_index, state_row in enumerate(request_rows):
        misses: list[tuple[int, int]] = []
        for query_offset, token_position in enumerate(query_indices[request_index]):
            if lookup_mask[request_index][query_offset] == 0:
                continue
            if token_position < 0 or token_position >= index_capacity:
                continue
            slot = state.index[state_row][token_position]
            if slot == INVALID_INDEX:
                misses.append((query_offset, token_position))
            else:
                slot_out[request_index][query_offset] = slot

        for miss_rank, (query_offset, token_position) in enumerate(misses):
            slot = state.free_slots[state_row][miss_rank]
            state.index[state_row][token_position] = slot
            state.slot_to_index[state_row][slot] = token_position
            slot_out[request_index][query_offset] = slot
            miss_out[request_index][query_offset] = 1

        if not misses:
            state.free_head[state_row][0] = 0
            continue

        protected_slots = {slot for slot in slot_out[request_index] if slot != INVALID_INDEX}
        cursor = state.free_head[state_row][1]
        victims = []
        for position in range(slot_count):
            slot = (cursor + position) % slot_count
            if slot not in protected_slots and state.slot_to_index[state_row][slot] != INVALID_INDEX:
                victims.append(slot)
                if len(victims) == len(misses):
                    break

        for victim_rank, slot in enumerate(victims):
            old_token = state.slot_to_index[state_row][slot]
            state.index[state_row][old_token] = INVALID_INDEX
            state.slot_to_index[state_row][slot] = INVALID_INDEX
            state.free_slots[state_row][len(misses) - 1 - victim_rank] = slot

        if victims:
            state.free_head[state_row][1] = (victims[-1] + 1) % slot_count
        state.free_head[state_row][0] = 0

    return slot_out, miss_out


def lookup_update_batch_reference(
    state: ReferenceState,
    request_rows: list[int],
    query_start_loc: list[int],
    query_indices: list[list[int]],
    lookup_mask: list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    query_width = len(query_indices[0])
    index_capacity = len(state.index[0])
    slot_count = len(state.slot_to_index[0])
    slot_out = [[INVALID_INDEX] * query_width for _ in query_indices]
    miss_out = [[0] * query_width for _ in query_indices]

    for request_index, state_row in enumerate(request_rows):
        protected_slots: set[int] = set()
        cursor = state.free_head[state_row][1]

        for query_index in range(query_start_loc[request_index], query_start_loc[request_index + 1]):
            misses: list[tuple[int, int]] = []
            for query_offset, token_position in enumerate(query_indices[query_index]):
                if lookup_mask[query_index][query_offset] == 0:
                    continue
                if token_position < 0 or token_position >= index_capacity:
                    continue
                slot = state.index[state_row][token_position]
                if slot == INVALID_INDEX:
                    slot_out[query_index][query_offset] = FALLBACK_SENTINEL
                    misses.append((query_offset, token_position))
                else:
                    slot_out[query_index][query_offset] = slot
                    protected_slots.add(slot)

            free_slots = [
                slot for slot in state.free_slots[state_row] if state.slot_to_index[state_row][slot] == INVALID_INDEX
            ]
            victims = [
                slot
                for position in range(slot_count)
                if (slot := (cursor + position) % slot_count) not in protected_slots
                and state.slot_to_index[state_row][slot] != INVALID_INDEX
            ]
            transaction_count = min(len(misses), len(free_slots), len(victims))

            for miss_rank, (query_offset, token_position) in enumerate(misses[:transaction_count]):
                slot = free_slots[miss_rank]
                state.index[state_row][token_position] = slot
                state.slot_to_index[state_row][slot] = token_position
                slot_out[query_index][query_offset] = slot
                miss_out[query_index][query_offset] = 1
                protected_slots.add(slot)

            reclaimed_slots = victims[:transaction_count]
            for victim_rank, slot in enumerate(reclaimed_slots):
                old_token = state.slot_to_index[state_row][slot]
                state.index[state_row][old_token] = INVALID_INDEX
                state.slot_to_index[state_row][slot] = INVALID_INDEX
                state.free_slots[state_row][transaction_count - 1 - victim_rank] = slot

            if reclaimed_slots:
                cursor = (reclaimed_slots[-1] + 1) % slot_count
                state.free_head[state_row][1] = cursor
            state.free_head[state_row][0] = 0

    return slot_out, miss_out
