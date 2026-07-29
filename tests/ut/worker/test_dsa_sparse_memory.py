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
from vllm_ascend.dsa_sparse_constants import DSA_SPARSE_QUERY_WIDTH
from vllm_ascend.worker.dsa_sparse_memory import (
    calculate_dsa_sparse_fixed_hbm_bytes,
    reserve_dsa_sparse_fixed_hbm_bytes,
)


def test_budget_covers_persistent_lookup_state_and_hot_cache():
    config = DSASparseCacheConfig(
        max_num_seqs=2,
        max_model_len=512,
        block_size=128,
        index_topk=DSA_SPARSE_QUERY_WIDTH,
    )
    layouts = (
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

    breakdown = calculate_dsa_sparse_fixed_hbm_bytes(
        config,
        layouts,
        cohort_count=2,
        backend_auxiliary_bytes=11,
    )

    assert breakdown.hot_payload_bytes == 352_512
    assert breakdown.lookup_state_bytes_per_cohort == 1_147_008
    assert breakdown.lookup_state_bytes == 2_294_016
    assert breakdown.core_fixed_tensor_bytes == 2_646_528
    assert breakdown.fixed_hbm_bytes == 2_646_539


def test_fixed_hbm_calculation_does_not_allocate_tensors(monkeypatch):
    def reject_tensor_allocation(*_args, **_kwargs):
        raise AssertionError("Fixed HBM calculation must not allocate tensors.")

    for name in ("empty", "zeros", "full", "arange", "tensor"):
        monkeypatch.setattr(torch, name, reject_tensor_allocation)

    config = DSASparseCacheConfig(
        max_num_seqs=1,
        max_model_len=512,
        block_size=128,
        index_topk=DSA_SPARSE_QUERY_WIDTH,
    )
    breakdown = calculate_dsa_sparse_fixed_hbm_bytes(
        config,
        (
            DSASparseLayerLayout(
                layer_name="layer.0",
                plane_dtypes=(torch.bfloat16,),
                plane_row_shapes=((1,),),
            ),
        ),
        cohort_count=1,
    )

    assert breakdown.fixed_hbm_bytes > 0


def test_breakdown_is_immutable():
    breakdown = calculate_dsa_sparse_fixed_hbm_bytes(
        DSASparseCacheConfig(
            max_num_seqs=1,
            max_model_len=512,
            block_size=128,
            index_topk=DSA_SPARSE_QUERY_WIDTH,
        ),
        (
            DSASparseLayerLayout(
                layer_name="layer.0",
                plane_dtypes=(torch.bfloat16,),
                plane_row_shapes=((1,),),
            ),
        ),
        cohort_count=1,
    )

    with pytest.raises(FrozenInstanceError):
        breakdown.cohort_count = 2


def test_reservation_deducts_fixed_hbm_and_reports_source():
    assert reserve_dsa_sparse_fixed_hbm_bytes(
        1000,
        400,
        source="test",
    ) == 600
    with pytest.raises(ValueError, match="test"):
        reserve_dsa_sparse_fixed_hbm_bytes(
            399,
            400,
            source="test",
        )


def test_zero_reservation_preserves_baseline_result():
    assert reserve_dsa_sparse_fixed_hbm_bytes(
        -1,
        0,
        source="baseline",
    ) == -1


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
