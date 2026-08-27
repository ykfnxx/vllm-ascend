# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch

from vllm_ascend.dsa_offload.constants import (
    FREE_HEAD_STRIDE,
    INDEX_CAPACITY,
    LOOKUP_SLOTS,
    REPLACEABLE_SLOTS,
    RESIDENT_SLOTS,
)
from vllm_ascend.dsa_offload.lookup import (
    IndexCacheCohort,
    clear_lookup_row,
    create_lookup_states,
    initialize_resident_mapping,
)


def test_lookup_state_has_fixed_shape_and_resident_mapping() -> None:
    cohort = IndexCacheCohort("leader", "leader", ("leader", "follower"), (0, 1))
    states = create_lookup_states((cohort,), 2, "cpu")
    state = states["leader"]

    assert state.index.shape == (2, INDEX_CAPACITY)
    assert state.slot_to_index.shape == (2, LOOKUP_SLOTS)
    assert state.free_slots.shape == (2, REPLACEABLE_SLOTS)
    assert state.free_head.shape == (2, FREE_HEAD_STRIDE)
    assert state.free_slots[0, :3].tolist() == [RESIDENT_SLOTS, RESIDENT_SLOTS + 1, RESIDENT_SLOTS + 2]

    initialize_resident_mapping(state, 1, [9, 3, 12])
    assert state.index[1, [9, 3, 12]].tolist() == [0, 1, 2]
    assert state.slot_to_index[1, :3].tolist() == [9, 3, 12]

    clear_lookup_row(states, 1)
    assert torch.count_nonzero(state.index[1] != -1) == 0
    assert state.free_head[1].tolist() == [0] * FREE_HEAD_STRIDE
