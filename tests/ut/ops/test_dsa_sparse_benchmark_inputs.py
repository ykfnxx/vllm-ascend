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
    seed = 1234
    inputs = COMMON.make_profile_inputs(
        _runtime(COMMON.LOOKUP_OPERATOR),
        requests=2,
        miss_count=miss_count,
        seed=seed,
    )

    assert inputs.query_index.shape == (2, COMMON.QUERY_COUNT)
    assert not torch.equal(
        inputs.query_index[0],
        inputs.query_index[1],
    )
    assert not torch.equal(
        inputs.slot_to_index[0, :COMMON.RESIDENT_SLOT_COUNT],
        inputs.slot_to_index[1, :COMMON.RESIDENT_SLOT_COUNT],
    )
    resident_slots = torch.arange(
        COMMON.RESIDENT_SLOT_COUNT,
        dtype=torch.int32,
    )
    for row in range(2):
        resident_positions = inputs.slot_to_index[
            row, :COMMON.RESIDENT_SLOT_COUNT
        ]
        assert (
            torch.unique(resident_positions).numel()
            == COMMON.RESIDENT_SLOT_COUNT
        )
        assert torch.equal(
            inputs.index[row].gather(
                0,
                resident_positions.long(),
            ),
            resident_slots,
        )
        query_row = inputs.query_index[row]
        assert torch.unique(query_row).numel() == COMMON.QUERY_COUNT
        assert int(query_row.min().item()) >= 0
        assert int(query_row.max().item()) < COMMON.INDEX_CAPACITY
        mapped_slots = inputs.index[row].gather(
            0,
            query_row.long(),
        )
        assert int(mapped_slots.eq(COMMON.INVALID_INDEX).sum()) == (
            miss_count
        )
    assert inputs.free_head[:, 0].tolist() == [0, 0]
    assert inputs.index.ne(COMMON.INVALID_INDEX).sum(dim=1).tolist() == [
        COMMON.RESIDENT_SLOT_COUNT,
        COMMON.RESIDENT_SLOT_COUNT,
    ]
    assert inputs.slot_to_index[
        :, COMMON.RESIDENT_SLOT_COUNT:
    ].eq(COMMON.INVALID_INDEX).all()


def test_lookup_workload_seed_is_reproducible() -> None:
    runtime = _runtime(COMMON.LOOKUP_OPERATOR)
    first = COMMON.make_profile_inputs(
        runtime,
        requests=2,
        miss_count=205,
        seed=7,
    )
    repeated = COMMON.make_profile_inputs(
        runtime,
        requests=2,
        miss_count=205,
        seed=7,
    )
    different = COMMON.make_profile_inputs(
        runtime,
        requests=2,
        miss_count=205,
        seed=8,
    )

    assert torch.equal(first.query_index, repeated.query_index)
    assert torch.equal(first.index, repeated.index)
    assert torch.equal(first.slot_to_index, repeated.slot_to_index)
    assert not torch.equal(first.query_index, different.query_index)
    assert not torch.equal(first.index, different.index)


@pytest.mark.parametrize("miss_count", (0, 205, 2048))
def test_maintain_workload_is_direct_post_lookup_state(
    miss_count: int,
) -> None:
    runtime = _runtime(COMMON.MAINTAIN_OPERATOR)
    seed = 1234
    lookup_inputs = COMMON.make_profile_inputs(
        runtime,
        requests=2,
        miss_count=miss_count,
        seed=seed,
    )
    inputs = COMMON.make_maintain_profile_inputs(
        runtime,
        requests=2,
        miss_count=miss_count,
        seed=seed,
    )

    assert inputs.free_head[:, 0].tolist() == [
        miss_count,
        miss_count,
    ]
    initial_slots = lookup_inputs.index.gather(
        1,
        lookup_inputs.query_index.long(),
    )
    if miss_count:
        miss_positions = lookup_inputs.query_index.masked_select(
            initial_slots.eq(COMMON.INVALID_INDEX)
        ).view(2, miss_count)
        allocated_slots = inputs.free_slots[:, :miss_count]
        assert torch.equal(
            inputs.index.gather(1, miss_positions.long()),
            allocated_slots,
        )
        assert torch.equal(
            inputs.slot_to_index.gather(
                1,
                allocated_slots.long(),
            ),
            miss_positions,
        )
    expected_query_slots = inputs.index.gather(
        1,
        lookup_inputs.query_index.long(),
    )
    assert torch.equal(inputs.last_query_slots, expected_query_slots)
    protected_tokens = inputs.slot_to_index.gather(
        1,
        inputs.last_query_slots.long(),
    )
    assert protected_tokens.ne(COMMON.INVALID_INDEX).all()
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
            seed=0,
        )
    with pytest.raises(ValueError, match="miss_count"):
        COMMON.make_profile_inputs(
            runtime,
            requests=1,
            miss_count=COMMON.QUERY_COUNT + 1,
            seed=0,
        )
