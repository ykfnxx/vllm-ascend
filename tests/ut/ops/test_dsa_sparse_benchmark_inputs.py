# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
COMMON_PATH = (
    ROOT
    / "tools"
    / "dsa_sparse_lookup_update"
    / "common.py"
)
SPEC = importlib.util.spec_from_file_location(
    "dsa_sparse_benchmark_common",
    COMMON_PATH,
)
assert SPEC is not None and SPEC.loader is not None
COMMON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMMON
SPEC.loader.exec_module(COMMON)


def _runtime(operator_name: str) -> object:
    return COMMON.Runtime(
        torch=torch,
        torch_npu=None,
        operator=None,
        operator_name=operator_name,
        device=torch.device("cpu"),
        install_root=None,
    )


@pytest.mark.parametrize("miss_count", (0, 205, 2048))
def test_lookup_workload_has_exact_requested_misses(
    miss_count: int,
) -> None:
    inputs = COMMON.make_profile_inputs(
        _runtime(COMMON.LOOKUP_OPERATOR),
        requests=2,
        miss_count=miss_count,
    )
    hit_count = COMMON.QUERY_COUNT - miss_count

    assert inputs.query_index.shape == (2, COMMON.QUERY_COUNT)
    assert torch.equal(
        inputs.query_index[0, :hit_count],
        torch.arange(hit_count, dtype=torch.int32),
    )
    assert torch.equal(
        inputs.query_index[0, hit_count:],
        torch.arange(
            COMMON.RESIDENT_SLOT_COUNT,
            COMMON.RESIDENT_SLOT_COUNT + miss_count,
            dtype=torch.int32,
        ),
    )
    assert inputs.free_head[:, 0].tolist() == [0, 0]
    assert (
        inputs.index[:, COMMON.RESIDENT_SLOT_COUNT :]
        .eq(COMMON.INVALID_INDEX)
        .all()
    )


@pytest.mark.parametrize("miss_count", (0, 205, 2048))
def test_maintain_workload_is_direct_post_lookup_state(
    miss_count: int,
) -> None:
    inputs = COMMON.make_maintain_profile_inputs(
        _runtime(COMMON.MAINTAIN_OPERATOR),
        requests=2,
        miss_count=miss_count,
    )
    hit_count = COMMON.QUERY_COUNT - miss_count
    allocated_slots = torch.arange(
        COMMON.RESIDENT_SLOT_COUNT,
        COMMON.RESIDENT_SLOT_COUNT + miss_count,
        dtype=torch.int32,
    )

    assert inputs.free_head[:, 0].tolist() == [
        miss_count,
        miss_count,
    ]
    assert torch.equal(
        inputs.last_query_slots[0, :hit_count],
        torch.arange(hit_count, dtype=torch.int32),
    )
    assert torch.equal(
        inputs.last_query_slots[0, hit_count:],
        allocated_slots,
    )
    assert torch.equal(
        inputs.index[
            0,
            COMMON.RESIDENT_SLOT_COUNT:
            COMMON.RESIDENT_SLOT_COUNT + miss_count,
        ],
        allocated_slots,
    )
    occupied = inputs.slot_to_index.ne(
        COMMON.INVALID_INDEX
    ).sum(dim=1)
    assert occupied.tolist() == [
        COMMON.RESIDENT_SLOT_COUNT + miss_count,
        COMMON.RESIDENT_SLOT_COUNT + miss_count,
    ]


def test_workload_rejects_miss_count_outside_query_width() -> None:
    runtime = _runtime(COMMON.LOOKUP_OPERATOR)

    with pytest.raises(ValueError, match="miss_count"):
        COMMON.make_profile_inputs(
            runtime,
            requests=1,
            miss_count=-1,
        )
    with pytest.raises(ValueError, match="miss_count"):
        COMMON.make_profile_inputs(
            runtime,
            requests=1,
            miss_count=COMMON.QUERY_COUNT + 1,
        )
