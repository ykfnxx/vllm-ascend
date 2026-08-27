# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest

from vllm_ascend.dsa_offload.constants import QUERY_WIDTH
from vllm_ascend.dsa_offload.pd import (
    DSAOffloadPDHandoff,
    build_handoff,
    validate_handoff,
)


def topk(offset: int = 0) -> list[int]:
    return list(range(offset, offset + QUERY_WIDTH))


def test_handoff_round_trip_preserves_rank_and_layer_data() -> None:
    handoff = build_handoff(
        request_id="request",
        stored_token_count=130,
        block_size=128,
        layer_topk_by_rank={
            0: {"layer.0": topk(), "layer.1": topk()},
            1: {"layer.0": topk(1), "layer.1": topk(1)},
        },
        partial_tail_blocks_by_rank={
            0: {"layer.0": 10, "layer.1": 11},
            1: {"layer.0": 12, "layer.1": 13},
        },
        tp_size=2,
    )

    assert DSAOffloadPDHandoff.from_dict(handoff.to_dict()) == handoff


def test_handoff_validates_width_layers_ranks_and_aligned_tail() -> None:
    with pytest.raises(ValueError, match="TopK must contain 2048"):
        build_handoff(
            request_id="request",
            stored_token_count=128,
            block_size=128,
            layer_topk_by_rank={0: {"layer": [1]}},
            partial_tail_blocks_by_rank={},
            tp_size=1,
        )

    with pytest.raises(ValueError, match="incomplete target layers"):
        build_handoff(
            request_id="request",
            stored_token_count=128,
            block_size=128,
            layer_topk_by_rank={0: {"a": topk()}, 1: {"b": topk()}},
            partial_tail_blocks_by_rank={},
            tp_size=2,
        )

    with pytest.raises(ValueError, match="must not contain partial tails"):
        build_handoff(
            request_id="request",
            stored_token_count=128,
            block_size=128,
            layer_topk_by_rank={0: {"layer": topk()}},
            partial_tail_blocks_by_rank={0: {"layer": 3}},
            tp_size=1,
        )

    with pytest.raises(ValueError, match="incomplete TP ranks"):
        build_handoff(
            request_id="request",
            stored_token_count=128,
            block_size=128,
            layer_topk_by_rank={1: {"layer": topk()}},
            partial_tail_blocks_by_rank={},
            tp_size=2,
        )


def test_decode_validates_deployment_layers_and_block_size() -> None:
    handoff = build_handoff(
        request_id="request",
        stored_token_count=128,
        block_size=128,
        layer_topk_by_rank={0: {"layer": topk()}},
        partial_tail_blocks_by_rank={},
        tp_size=1,
    )

    with pytest.raises(ValueError, match="block sizes differ"):
        validate_handoff(handoff, 1, ("layer",), 64)
    with pytest.raises(ValueError, match="incomplete target layers"):
        validate_handoff(handoff, 1, ("layer", "missing"), 128)
