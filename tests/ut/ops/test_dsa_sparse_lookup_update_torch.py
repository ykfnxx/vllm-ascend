# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

from vllm_ascend.attention.dsa_sparse import (
    DSASparseCacheConfig,
    DSASparseCohortKey,
    DSASparsePlan,
    DSASparsePlanKey,
    DSASparseResidencyState,
)
from vllm_ascend.ops.dsa_sparse import (
    DSASparseLookupUpdateTorchOperator,
)


@patch(
    "torch.ops._C_ascend.dsa_sparse_lookup_update",
    create=True,
)
def test_torch_wrapper_forwards_the_frozen_16_tensor_abi(mock_operator):
    config = DSASparseCacheConfig(
        max_num_seqs=2,
        max_model_len=32,
        block_size=4,
        device_buffer_size=8,
        max_query_tokens_per_request=1,
        index_topk=4,
    )
    cohort_key = DSASparseCohortKey(
        name="target-indexer-0",
        role="target",
    )
    state = DSASparseResidencyState.allocate(
        config,
        cohort_key,
        device="cpu",
    )
    plan = DSASparsePlan.allocate(
        config,
        DSASparsePlanKey(
            token_capacity=2,
            request_capacity=2,
            query_lane_capacity=1,
            role="target",
        ),
        device="cpu",
    )

    DSASparseLookupUpdateTorchOperator().lookup_update(
        state=state,
        plan=plan,
    )

    mock_operator.assert_called_once_with(
        state.token_to_hot,
        state.hot_to_token,
        state.lru_slots,
        state.state_seat_epoch,
        plan.row_mapping.row_to_cache_seat,
        plan.row_mapping.row_seat_epoch,
        plan.query_positions,
        plan.query_to_row,
        plan.query_to_lane,
        plan.query_valid_mask,
        plan.valid_topk_counts,
        plan.seq_lens,
        plan.topk_positions,
        plan.resolved_hot_indices,
        plan.miss_mask,
        plan.workspace,
    )
