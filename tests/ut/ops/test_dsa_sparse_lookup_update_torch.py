# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import Mock, call, patch

import torch

from vllm_ascend.attention.dsa_sparse import (
    DSASparseCohortKey,
    DSASparseLookupBatch,
    DSASparseLookupState,
)
from vllm_ascend.ops.dsa_sparse import (
    TorchDSASparseLookupOperator,
)


def _make_call():
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
    return state, batch, expected_slots, expected_misses


def test_torch_operator_passes_the_frozen_asu_shaped_abi() -> None:
    state, batch, expected_slots, expected_misses = _make_call()

    with patch(
        "torch.ops._C_ascend.dsa_sparse_lookup_update",
        create=True,
        return_value=(expected_slots, expected_misses),
    ) as custom_op, patch(
        "vllm_ascend.ops.dsa_sparse.dsa_sparse_probe.is_enabled",
        return_value=False,
    ):
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


def test_torch_operator_can_select_asu_hbm_index_lookup() -> None:
    state, batch, expected_slots, expected_misses = _make_call()

    with patch(
        "torch.ops._C_ascend.asu_hbm_index_lookup",
        create=True,
        return_value=(expected_slots, expected_misses),
    ) as lookup_op, patch(
        "torch.ops._C_ascend.asu_hbm_index_maintain_aicpu",
        create=True,
    ) as maintain_op, patch(
        "torch.npu.synchronize",
        create=True,
    ) as synchronize, patch(
        "vllm_ascend.ops.dsa_sparse.dsa_sparse_probe.is_enabled",
        return_value=False,
    ):
        calls = Mock()
        calls.attach_mock(lookup_op, "lookup")
        calls.attach_mock(maintain_op, "maintain")
        calls.attach_mock(synchronize, "synchronize")
        output = TorchDSASparseLookupOperator(
            lookup_backend="asu_hbm_index_lookup"
        ).lookup(
            state=state,
            batch=batch,
        )

    assert calls.mock_calls == [
        call.lookup(
            state.index,
            state.slot_to_index,
            state.free_slots,
            state.free_head,
            batch.req_pool_entries,
            batch.query_index,
            batch.lookup_mask,
            2,
        ),
        call.maintain(
            state.index,
            state.slot_to_index,
            state.free_slots,
            state.free_head,
            batch.req_pool_entries,
            expected_slots,
            2,
            0,
        ),
        call.synchronize(),
    ]
    assert output.slot_out is expected_slots
    assert output.miss_out is expected_misses


def test_torch_operator_probe_records_new_input_output_contract() -> None:
    state, batch, expected_slots, expected_misses = _make_call()

    with patch(
        "torch.ops._C_ascend.dsa_sparse_lookup_update",
        create=True,
        return_value=(expected_slots, expected_misses),
    ), patch(
        "vllm_ascend.ops.dsa_sparse.dsa_sparse_probe.is_enabled",
        return_value=True,
    ), patch(
        "vllm_ascend.ops.dsa_sparse.dsa_sparse_probe.synchronize_device",
    ) as synchronize, patch(
        "vllm_ascend.ops.dsa_sparse.dsa_sparse_probe.emit",
    ) as emit:
        TorchDSASparseLookupOperator().lookup(
            state=state,
            batch=batch,
        )

    synchronize.assert_called_once_with()
    emit.assert_called_once_with(
        "lookup_update_done",
        cohort="cohort",
        role="target",
        req_num=2,
        req_pool_entries_shape=[2],
        query_index_shape=[2, 2],
        lookup_mask_shape=[2, 2],
        slot_out_shape=[2, 2],
        miss_out_shape=[2, 2],
    )
