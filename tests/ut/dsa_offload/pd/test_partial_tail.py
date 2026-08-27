# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from vllm_ascend.dsa_offload.constants import QUERY_WIDTH
from vllm_ascend.dsa_offload.pd import (
    DSAOffloadPDHandoff,
    append_partial_tail_transfer,
)


def make_handoff(stored_tokens: int) -> DSAOffloadPDHandoff:
    return DSAOffloadPDHandoff(
        remote_request_id="request",
        stored_token_count=stored_tokens,
        block_size=4,
        layer_topk_by_rank={0: {"layer": [0] * QUERY_WIDTH}},
        partial_tail_blocks_by_rank=({0: {"layer": 7}} if stored_tokens % 4 else {}),
    )


def test_partial_tail_uses_remote_source_local_hot_tail_and_valid_bytes() -> None:
    local = {
        "layer": [
            {
                "base_addr": 1000,
                "block_stride": 40,
                "token_bytes": 10,
                "hot_block_base": 100,
                "hot_blocks_per_row": 20,
                "tail_block_offset": 8,
            }
        ]
    }
    remote = {
        "layer": [
            {
                "base_addr": 5000,
                "block_stride": 40,
            }
        ]
    }
    local_addresses: list[int] = []
    remote_addresses: list[int] = []
    lengths: list[int] = []

    append_partial_tail_transfer(
        handoff=make_handoff(6),
        tp_rank=0,
        row_id=2,
        local_regions=local,
        remote_regions=remote,
        local_addresses=local_addresses,
        remote_addresses=remote_addresses,
        lengths=lengths,
    )

    assert local_addresses == [1000 + (100 + 2 * 20 + 8) * 40]
    assert remote_addresses == [5000 + 7 * 40]
    assert lengths == [2 * 10]


def test_block_aligned_tail_is_noop() -> None:
    local_addresses = [1]
    remote_addresses = [2]
    lengths = [3]

    append_partial_tail_transfer(
        handoff=make_handoff(8),
        tp_rank=0,
        row_id=0,
        local_regions={},
        remote_regions={},
        local_addresses=local_addresses,
        remote_addresses=remote_addresses,
        lengths=lengths,
    )

    assert (local_addresses, remote_addresses, lengths) == ([1], [2], [3])
