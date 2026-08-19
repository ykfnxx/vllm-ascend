# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.worker.dsa_sparse_memory import (
    calculate_dsa_sparse_fixed_hbm_bytes,
    reserve_dsa_sparse_fixed_hbm_bytes,
)


def test_budget_covers_persistent_lookup_state_and_hot_cache():
    layouts = (
        (torch.bfloat16, 2, 3),
        (torch.uint8, 1, 5),
    )

    breakdown = calculate_dsa_sparse_fixed_hbm_bytes(
        2,
        128,
        layouts,
        cohort_count=2,
    )

    assert breakdown.hot_payload_bytes == 352_512
    assert breakdown.lookup_state_bytes_per_cohort == 1_147_008
    assert breakdown.lookup_state_bytes == 2_294_016
    assert breakdown.core_fixed_tensor_bytes == 2_646_528
    assert breakdown.fixed_hbm_bytes == 2_646_528
    assert breakdown.lookup_capacity == 10_240
    assert breakdown.transient_region_span == 128
    assert breakdown.fallback_slot_count == 0
    assert breakdown.verify_staging_capacity == 0


def test_mtp_verify_staging_reuses_existing_tail_block():
    baseline = calculate_dsa_sparse_fixed_hbm_bytes(
        2,
        128,
        ((torch.bfloat16, 2, 3), (torch.uint8, 1, 5)),
        cohort_count=2,
    )
    mtp = calculate_dsa_sparse_fixed_hbm_bytes(
        2,
        128,
        ((torch.bfloat16, 2, 3), (torch.uint8, 1, 5)),
        cohort_count=2,
        max_verify_tokens_per_request=16,
        uses_mtp=True,
    )

    assert mtp.hot_payload_bytes == baseline.hot_payload_bytes
    assert mtp.fixed_hbm_bytes == baseline.fixed_hbm_bytes
    assert mtp.transient_region_span == 128
    assert mtp.fallback_slot_count == 1
    assert mtp.verify_staging_capacity == 16


def test_mtp_verify_staging_grows_by_aligned_blocks_only():
    mtp = calculate_dsa_sparse_fixed_hbm_bytes(
        2,
        128,
        ((torch.bfloat16, 2, 3), (torch.uint8, 1, 5)),
        cohort_count=2,
        max_verify_tokens_per_request=128,
        uses_mtp=True,
    )

    assert mtp.transient_region_span == 256
    assert mtp.hot_payload_bytes == 356_864


def test_fixed_hbm_calculation_does_not_allocate_tensors(monkeypatch):
    def reject_tensor_allocation(*_args, **_kwargs):
        raise AssertionError("Fixed HBM calculation must not allocate tensors.")

    for name in ("empty", "zeros", "full", "arange", "tensor"):
        monkeypatch.setattr(torch, name, reject_tensor_allocation)

    breakdown = calculate_dsa_sparse_fixed_hbm_bytes(
        1,
        128,
        ((torch.bfloat16, 1, 1),),
        cohort_count=1,
    )

    assert breakdown.fixed_hbm_bytes > 0


def test_breakdown_is_immutable():
    breakdown = calculate_dsa_sparse_fixed_hbm_bytes(
        1,
        128,
        ((torch.bfloat16, 1, 1),),
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
