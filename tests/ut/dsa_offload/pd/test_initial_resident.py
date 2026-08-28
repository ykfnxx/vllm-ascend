# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.dsa_offload.constants import QUERY_WIDTH, RESIDENT_SLOTS
from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout, HotCacheState
from vllm_ascend.dsa_offload.lookup import (
    IndexCacheCohort,
    create_lookup_states,
)
from vllm_ascend.dsa_offload.pd import (
    DSAOffloadLocalHandoff,
    DSAOffloadPDHandoff,
    admit_from_handoff,
    admit_local_from_prefill,
    select_initial_resident,
)


def test_topk_first_deduplicates_filters_and_fills_earliest_history() -> None:
    selected = select_initial_resident(
        [9000, -1, 5, 5, 4],
        stored_token_count=8300,
        block_size=128,
    )

    assert len(selected) == RESIDENT_SLOTS
    assert selected[:4] == [5, 4, 0, 1]
    assert len(set(selected)) == len(selected)
    assert max(selected) < 8192


def test_admission_loads_each_follower_before_exposing_mapping(spy_io) -> None:
    layout = HotCacheLayout(4, 1, 2)
    caches = {
        "leader": (torch.empty((layout.hot_blocks, 4, 1)),),
        "follower": (torch.empty((layout.hot_blocks, 4, 1)),),
    }
    hot_cache = HotCacheState(layout, caches)
    hot_cache.admit("decode")
    cohort = IndexCacheCohort(
        "leader",
        "leader",
        ("leader", "follower"),
        (3, 4),
    )
    states = create_lookup_states((cohort,), 1, "cpu")
    positions = list(range(4)) + [-1] * (QUERY_WIDTH - 4)
    handoff = DSAOffloadPDHandoff(
        remote_request_id="prefill",
        stored_token_count=4,
        block_size=4,
        layer_topk_by_rank={
            0: {"leader": positions, "follower": positions},
        },
        partial_tail_blocks_by_rank={},
    )

    original_get = spy_io.get_tokens

    def get_before_mapping(**kwargs):
        assert states["leader"].index[0, :4].tolist() == [-1, -1, -1, -1]
        original_get(**kwargs)

    spy_io.get_tokens = get_before_mapping
    admit_from_handoff(
        request_id="decode",
        handoff=handoff,
        tp_rank=0,
        hot_cache=hot_cache,
        cohorts=(cohort,),
        lookup_states=states,
        layer_ids={"leader": 3, "follower": 4},
        committed_block_hashes=[b"block-0"],
        io_backend=spy_io,
    )

    assert [call["layer_id"] for call in spy_io.get_calls] == [3, 4]
    assert all(call["destination_slots"].tolist() == [0, 1, 2, 3] for call in spy_io.get_calls)
    assert states["leader"].index[0, :4].tolist() == [0, 1, 2, 3]
    assert "decode" in hot_cache.ready_requests


def test_local_admission_loads_full_history_and_copies_partial_tail(
    spy_io,
) -> None:
    layout = HotCacheLayout(4, 1, 2, hot_block_base=4)
    plane = torch.zeros((4 + layout.hot_blocks, 4, 1))
    plane[1, 0] = 37
    hot_cache = HotCacheState(layout, {"layer": (plane,)})
    cohort = IndexCacheCohort("layer", "layer", ("layer",), (3,))
    states = create_lookup_states((cohort,), 1, "cpu")
    positions = list(range(4)) + [-1] * (QUERY_WIDTH - 4)
    handoff = DSAOffloadLocalHandoff(
        request_id="request",
        stored_token_count=5,
        block_size=4,
        layer_topk={"layer": positions},
        partial_tail_blocks={"layer": 1},
    )

    row_id = admit_local_from_prefill(
        handoff=handoff,
        hot_cache=hot_cache,
        cohorts=(cohort,),
        lookup_states=states,
        layer_ids={"layer": 3},
        committed_block_hashes=[b"block-0"],
        io_backend=spy_io,
    )

    tail_block = layout.row_block_base(row_id) + layout.tail_block_offset
    assert plane[tail_block, 0].item() == 37
    assert spy_io.get_calls[0]["destination_slots"].tolist() == [
        layout.global_slot(row_id, offset) for offset in range(4)
    ]
    assert states["layer"].index[row_id, :4].tolist() == [0, 1, 2, 3]
    assert "request" in hot_cache.ready_requests


def test_local_admission_rejects_missing_full_block_hashes(spy_io) -> None:
    layout = HotCacheLayout(4, 1, 2, hot_block_base=4)
    plane = torch.zeros((4 + layout.hot_blocks, 4, 1))
    hot_cache = HotCacheState(layout, {"layer": (plane,)})
    cohort = IndexCacheCohort("layer", "layer", ("layer",), (3,))
    states = create_lookup_states((cohort,), 1, "cpu")
    handoff = DSAOffloadLocalHandoff(
        request_id="request",
        stored_token_count=4,
        block_size=4,
        layer_topk={
            "layer": list(range(4)) + [-1] * (QUERY_WIDTH - 4),
        },
        partial_tail_blocks={},
    )

    with pytest.raises(
        RuntimeError,
        match=r"Hot Cache admission for request request requires 1 block hashes",
    ):
        admit_local_from_prefill(
            handoff=handoff,
            hot_cache=hot_cache,
            cohorts=(cohort,),
            lookup_states=states,
            layer_ids={"layer": 3},
            committed_block_hashes=[],
            io_backend=spy_io,
        )
