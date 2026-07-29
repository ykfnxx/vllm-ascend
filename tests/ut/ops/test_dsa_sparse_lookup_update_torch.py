# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

import torch

from vllm_ascend.attention.dsa_sparse import (
    DSASparseCohortKey,
    DSASparseLookupBatch,
    DSASparseLookupState,
)
from vllm_ascend.ops.dsa_sparse import (
    TorchDSASparseLookupOperator,
)


def test_torch_operator_passes_the_frozen_asu_shaped_abi() -> None:
    state = DSASparseLookupState(
        cohort=DSASparseCohortKey("cohort", "target"),
        index=torch.empty((3, 1), dtype=torch.int32),
        slot_to_index=torch.empty((3, 1), dtype=torch.int32),
        free_slots=torch.empty((3, 1), dtype=torch.int32),
        free_head=torch.empty((3, 2), dtype=torch.int32),
    )
    batch = DSASparseLookupBatch(
        req_pool_entries=torch.tensor([2, 0], dtype=torch.int32),
        query_index=torch.tensor([[4, 5], [6, 7]], dtype=torch.int32),
        lookup_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.int32),
    )
    expected_slots = torch.tensor([[1, 2], [3, -1]], dtype=torch.int32)
    expected_misses = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    with patch(
        "torch.ops._C_ascend.dsa_sparse_lookup_update",
        create=True,
        return_value=(expected_slots, expected_misses),
    ) as custom_op:
        output = TorchDSASparseLookupOperator().lookup(
            state=state,
            batch=batch,
        )

    custom_op.assert_called_once_with(
        state.index,
        state.slot_to_index,
        state.free_slots,
        state.free_head,
        batch.req_pool_entries,
        batch.query_index,
        batch.lookup_mask,
        2,
    )
    assert output.slot_out is expected_slots
    assert output.miss_out is expected_misses
