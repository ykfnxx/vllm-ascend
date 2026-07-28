# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.attention.dsa_sparse import (
    DSASparseCacheConfig,
    DSASparseLayerLayout,
)
from vllm_ascend.worker.dsa_sparse_memory import (
    calculate_dsa_sparse_fixed_hbm_bytes,
    reserve_dsa_sparse_fixed_hbm_bytes,
)


def _config() -> DSASparseCacheConfig:
    return DSASparseCacheConfig(
        max_num_seqs=2,
        max_model_len=10,
        block_size=4,
        device_buffer_size=7,
        max_query_tokens_per_request=2,
        index_topk=3,
    )


def _layouts() -> tuple[DSASparseLayerLayout, ...]:
    return (
        DSASparseLayerLayout(
            layer_name="layer.0",
            plane_dtypes=(torch.bfloat16,),
            plane_row_shapes=((2, 3),),
        ),
        DSASparseLayerLayout(
            layer_name="layer.1",
            plane_dtypes=(torch.uint8,),
            plane_row_shapes=((5,),),
        ),
    )


def test_calculates_exact_fixed_tensor_bytes_without_tensor_allocation(
    monkeypatch,
):
    def reject_tensor_allocation(*_args, **_kwargs):
        raise AssertionError("The fixed HBM calculator must not allocate tensors.")

    for name in ("empty", "zeros", "full", "arange", "tensor"):
        monkeypatch.setattr(torch, name, reject_tensor_allocation)

    breakdown = calculate_dsa_sparse_fixed_hbm_bytes(
        _config(),
        _layouts(),
        cohort_count=2,
        max_sfa_queries=8,
        backend_auxiliary_bytes=11,
    )

    # 24 hot rows * (6 BF16 + 5 UINT8 bytes per row).
    assert breakdown.hot_payload_bytes == 408
    # Role-level request/query/block/newest descriptors are shared by cohorts.
    assert breakdown.batch_metadata_bytes == 144
    # token_to_hot + hot_to_token + LRU.
    assert breakdown.residency_state_bytes_per_cohort == 192
    # Lookup input/output tensors plus the fused operator workspace.
    assert breakdown.lookup_plan_bytes_per_cohort == 6468
    # max(flat query initializer[Q=4], LRU initializer[S=7]).
    assert breakdown.initialization_scratch_bytes == 28
    # Context: active query/request indices + padded SFA buffers.
    assert breakdown.eager_context_bytes_per_cohort == 176
    # One shared vectorized metadata-staging pass.
    assert breakdown.eager_batch_staging_bytes == 280
    # Lookup: fixed top-k/count plus the larger resolved-index gather.
    assert breakdown.eager_lookup_scratch_bytes_per_cohort == 112
    assert breakdown.eager_execution_reserve_bytes_per_cohort == 288
    assert breakdown.residency_state_bytes == 384
    assert breakdown.lookup_plan_bytes == 12936
    assert breakdown.core_fixed_tensor_bytes == 13872
    assert breakdown.eager_execution_reserve_bytes == 856
    assert breakdown.runtime_peak_reserve_bytes == 856
    assert breakdown.fixed_hbm_bytes == 14739


def test_initialization_scratch_can_define_the_runtime_peak():
    config = DSASparseCacheConfig(
        max_num_seqs=1,
        max_model_len=128,
        block_size=128,
        device_buffer_size=4096,
        max_query_tokens_per_request=1,
        index_topk=1,
    )
    breakdown = calculate_dsa_sparse_fixed_hbm_bytes(
        config,
        (
            DSASparseLayerLayout(
                layer_name="layer.0",
                plane_dtypes=(torch.uint8,),
                plane_row_shapes=((1,),),
            ),
        ),
        cohort_count=1,
        max_sfa_queries=1,
    )

    assert breakdown.initialization_scratch_bytes == 4096 * 4
    assert (
        breakdown.initialization_scratch_bytes
        > breakdown.eager_execution_reserve_bytes
    )
    assert (
        breakdown.runtime_peak_reserve_bytes
        == breakdown.initialization_scratch_bytes
    )
    assert breakdown.fixed_hbm_bytes == (
        breakdown.core_fixed_tensor_bytes
        + breakdown.initialization_scratch_bytes
    )


def test_breakdown_is_immutable():
    breakdown = calculate_dsa_sparse_fixed_hbm_bytes(
        _config(),
        _layouts(),
        cohort_count=1,
        max_sfa_queries=8,
    )

    with pytest.raises(FrozenInstanceError):
        breakdown.cohort_count = 2


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"cohort_count": 0}, ValueError, "cohort_count must be positive"),
        ({"cohort_count": True}, TypeError, "cohort_count must be an integer"),
        (
            {"cohort_count": 1, "max_sfa_queries": 0},
            ValueError,
            "max_sfa_queries must be positive",
        ),
        (
            {"cohort_count": 1, "backend_auxiliary_bytes": -1},
            ValueError,
            "backend_auxiliary_bytes must be non-negative",
        ),
    ),
)
def test_rejects_invalid_fixed_budget_inputs(kwargs, error, message):
    calculation_kwargs = {
        "max_sfa_queries": 8,
        **kwargs,
    }
    with pytest.raises(error, match=message):
        calculate_dsa_sparse_fixed_hbm_bytes(
            _config(),
            _layouts(),
            **calculation_kwargs,
        )


def test_rejects_empty_or_duplicate_layer_layouts():
    with pytest.raises(ValueError, match="At least one local"):
        calculate_dsa_sparse_fixed_hbm_bytes(
            _config(),
            (),
            cohort_count=1,
            max_sfa_queries=8,
        )

    duplicate = DSASparseLayerLayout(
        layer_name="layer.0",
        plane_dtypes=(torch.float32,),
        plane_row_shapes=((1,),),
    )
    with pytest.raises(ValueError, match="unique names"):
        calculate_dsa_sparse_fixed_hbm_bytes(
            _config(),
            (*_layouts(), duplicate),
            cohort_count=1,
            max_sfa_queries=8,
        )


def test_reservation_deducts_once_and_fails_if_it_does_not_fit():
    assert (
        reserve_dsa_sparse_fixed_hbm_bytes(
            1_000,
            400,
            source="test budget",
        )
        == 600
    )
    with pytest.raises(ValueError, match="test budget"):
        reserve_dsa_sparse_fixed_hbm_bytes(
            399,
            400,
            source="test budget",
        )


def test_zero_reservation_preserves_baseline_negative_result():
    assert (
        reserve_dsa_sparse_fixed_hbm_bytes(
            -1,
            0,
            source="baseline",
        )
        == -1
    )


def test_worker_uses_model_runner_budget_provider_for_decode_consumer():
    from vllm_ascend.worker.worker import NPUWorker

    worker = NPUWorker.__new__(NPUWorker)
    worker.dsa_sparse_config = SimpleNamespace(is_producer=False)
    worker.dsa_sparse_fixed_hbm_bytes = None
    provider = MagicMock(return_value=1234)
    worker.model_runner = SimpleNamespace(
        get_dsa_sparse_fixed_hbm_bytes=provider,
    )

    assert worker.get_dsa_sparse_fixed_hbm_bytes() == 1234
    provider.assert_called_once_with()


def test_worker_explicit_budget_binding_overrides_provider_once():
    from vllm_ascend.worker.worker import NPUWorker

    worker = NPUWorker.__new__(NPUWorker)
    worker.dsa_sparse_config = SimpleNamespace(is_producer=False)
    worker.dsa_sparse_fixed_hbm_bytes = None
    provider = MagicMock(return_value=1234)
    worker.model_runner = SimpleNamespace(
        get_dsa_sparse_fixed_hbm_bytes=provider,
    )

    worker.bind_dsa_sparse_fixed_hbm_bytes(4321)

    assert worker.get_dsa_sparse_fixed_hbm_bytes() == 4321
    provider.assert_not_called()
    with pytest.raises(RuntimeError, match="already bound"):
        worker.bind_dsa_sparse_fixed_hbm_bytes(5678)
