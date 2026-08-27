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
    query_indices = torch.empty((2, 5), dtype=torch.int32)
    lookup_mask = torch.empty((2, 5), dtype=torch.int32)
    output = (torch.empty_like(query_indices), torch.empty_like(query_indices))

    with patch(
        "torch.ops._C_ascend.dsa_offload_lookup_update",
        return_value=output,
        create=True,
    ) as operator:
        result = lookup_update(state, request_rows, query_indices, lookup_mask)

    operator.assert_called_once_with(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        request_rows,
        query_indices,
        lookup_mask,
        2,
    )
    assert result is output
