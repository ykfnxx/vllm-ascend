# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout, HotCacheState


def test_admission_release_abort_and_row_reuse_clear_transient() -> None:
    layout = HotCacheLayout(128, 2, 8)
    cache = torch.ones((layout.hot_blocks, layout.block_size, 1))
    state = HotCacheState(layout, {"layer": (cache,)})

    assert state.admit("first") == 0
    transient = cache.flatten(0, 1)[layout.tail_base : layout.row_stride]
    assert torch.count_nonzero(transient) == 0

    state.mark_ready("first")
    transient.fill_(3)
    state.release("first")
    assert "first" not in state.ready_requests
    assert torch.count_nonzero(transient) == 0

    assert state.admit("second") == 1
    state.release("second")
    assert state.admit("reused") == 0


def test_decode_preemption_fails_without_releasing_row() -> None:
    layout = HotCacheLayout(128, 1, 1)
    cache = torch.empty((layout.hot_blocks, layout.block_size, 1))
    state = HotCacheState(layout, {"layer": (cache,)})
    state.admit("decode")

    with pytest.raises(RuntimeError, match="does not support Decode preemption"):
        state.fail_on_preemption({"decode"})

    assert state.request_to_row == {"decode": 0}
