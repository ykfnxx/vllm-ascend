# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

import torch

from vllm_ascend.ops.dsa_sparse import dsa_sparse_lookup_update_batch


def test_batch_lookup_wrapper_forwards_the_packed_operator_abi():
    index = torch.empty((4, 16), dtype=torch.int32)
    slot_to_index = torch.empty((4, 8), dtype=torch.int32)
    free_slots = torch.empty((4, 2), dtype=torch.int32)
    free_head = torch.empty((4, 16), dtype=torch.int32)
    req_pool_entries = torch.tensor([3, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    query_index = torch.empty((5, 5), dtype=torch.int32)
    lookup_mask = torch.empty((5, 5), dtype=torch.int32)
    slot_out = torch.empty_like(query_index)
    miss_out = torch.empty_like(query_index)

    with patch(
        "torch.ops._C_ascend.dsa_sparse_lookup_update_batch",
        return_value=(slot_out, miss_out),
    ) as op:
        result = dsa_sparse_lookup_update_batch(
            index,
            slot_to_index,
            free_slots,
            free_head,
            req_pool_entries,
            query_start_loc,
            query_index,
            lookup_mask,
        )

    op.assert_called_once_with(
        index,
        slot_to_index,
        free_slots,
        free_head,
        req_pool_entries,
        query_start_loc,
        query_index,
        lookup_mask,
        2,
    )
    assert result[0] is slot_out
    assert result[1] is miss_out
