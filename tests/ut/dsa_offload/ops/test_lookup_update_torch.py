# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

import torch

from vllm_ascend.dsa_offload.ops import LookupState, lookup_update


def test_lookup_wrapper_forwards_operator_abi() -> None:
    state = LookupState(
        index=torch.empty((4, 16), dtype=torch.int32),
        slot_to_index=torch.empty((4, 8), dtype=torch.int32),
        free_slots=torch.empty((4, 2), dtype=torch.int32),
        free_head=torch.empty((4, 16), dtype=torch.int32),
    )
    request_rows = torch.tensor([3, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32)
    query_positions = torch.tensor([8, 12, 13], dtype=torch.int64)
    semantic_topk = torch.empty((3, 1, 2048), dtype=torch.int32)
    output = (
        torch.empty_like(semantic_topk),
        torch.empty_like(semantic_topk),
    )

    with patch(
        "torch.ops._C_ascend.dsa_offload_lookup_update",
        return_value=output,
        create=True,
    ) as operator:
        result = lookup_update(
            state,
            request_rows,
            query_start_loc,
            query_positions,
            semantic_topk,
            block_size=128,
            tail_base=10240,
            fallback_slot=10368,
            staging_base=10369,
            decode_mode=1,
        )

    operator.assert_called_once_with(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        request_rows,
        query_start_loc,
        query_positions,
        semantic_topk,
        2,
        128,
        10240,
        10368,
        10369,
        1,
    )
    assert result is output
