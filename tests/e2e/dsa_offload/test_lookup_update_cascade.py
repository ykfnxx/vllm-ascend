# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.dsa_offload.ops import LookupState, lookup_update
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_custom_op,
    get_ascend_device_type,
)

enable_custom_op()

pytestmark = [
    pytest.mark.ascend_a5,
    pytest.mark.skipif(
        get_ascend_device_type() != AscendDeviceType.A5,
        reason="DsaOffloadLookupUpdate requires Ascend A5",
    ),
]

INDEX_CAPACITY = 128 * 1024
RESIDENT_SLOTS = 8 * 1024
LOOKUP_SLOTS = 10 * 1024
FREE_SLOTS = 2 * 1024


def make_state(rows: int) -> LookupState:
    index = torch.full(
        (rows, INDEX_CAPACITY), -1, dtype=torch.int32, device="npu"
    )
    slot_to_index = torch.full(
        (rows, LOOKUP_SLOTS), -1, dtype=torch.int32, device="npu"
    )
    resident = torch.arange(4, dtype=torch.int32, device="npu")
    index[:, :4] = resident
    slot_to_index[:, :4] = resident
    return LookupState(
        index=index,
        slot_to_index=slot_to_index,
        free_slots=torch.arange(
            RESIDENT_SLOTS,
            LOOKUP_SLOTS,
            dtype=torch.int32,
            device="npu",
        )
        .expand(rows, FREE_SLOTS)
        .clone(),
        free_head=torch.zeros((rows, 16), dtype=torch.int32, device="npu"),
    )


def run_lookup(
    state: LookupState,
    request_rows: list[int],
    query_start_loc: list[int],
    query_positions: list[int],
    semantic_topk: torch.Tensor,
    decode_mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return lookup_update(
        state,
        torch.tensor(request_rows, dtype=torch.int32, device="npu"),
        torch.tensor(query_start_loc, dtype=torch.int32, device="npu"),
        torch.tensor(query_positions, dtype=torch.int64, device="npu"),
        semantic_topk,
        block_size=4,
        tail_base=10240,
        fallback_slot=10244,
        staging_base=10245,
        decode_mode=decode_mode,
    )


@torch.inference_mode()
def test_normal_decode_and_prefill_passthrough() -> None:
    state = make_state(2)
    semantic = torch.full(
        (2, 1, 2048), -1, dtype=torch.int32, device="npu"
    )
    semantic[0, 0, :3] = torch.tensor([3, 2, 1], device="npu")
    semantic[1, 0, :5] = torch.tensor([1, 6, 8, 9, -1], device="npu")

    mapped, misses = run_lookup(
        state,
        [-1, 0],
        [0, 1, 2],
        [3, 8],
        semantic,
        decode_mode=0,
    )

    assert torch.equal(mapped[0], semantic[0])
    assert mapped[1, 0, :5].cpu().tolist() == [
        1,
        8192,
        10240,
        -1,
        -1,
    ]
    assert misses[1, 0, :5].cpu().tolist() == [0, 1, 0, 0, 0]
    assert state.index[0, 6].item() == 8192


@torch.inference_mode()
def test_mtp_tail_staging_and_cross_query_protection() -> None:
    state = make_state(1)
    semantic = torch.full(
        (2, 1, 2048), -1, dtype=torch.int32, device="npu"
    )
    semantic[0, 0, :4] = torch.tensor([1, 6, 8, -1], device="npu")
    semantic[1, 0, :4] = torch.tensor([6, 7, 8, 9], device="npu")

    mapped, misses = run_lookup(
        state,
        [0],
        [0, 2],
        [8, 9],
        semantic,
        decode_mode=1,
    )

    assert mapped[:, 0, :4].cpu().tolist() == [
        [1, 8192, 10245, -1],
        [8192, 0, 10245, 10246],
    ]
    assert misses[:, 0, :4].cpu().tolist() == [
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ]
    assert state.index[0, [1, 6, 7]].cpu().tolist() == [1, 8192, 0]
