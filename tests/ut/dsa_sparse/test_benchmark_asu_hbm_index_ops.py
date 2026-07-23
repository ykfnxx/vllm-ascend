from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "dsa_sparse"
    / "benchmark_asu_hbm_index_ops.py"
)
SPEC = importlib.util.spec_from_file_location(
    "benchmark_asu_hbm_index_ops", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


class FakeTensor:

    def __init__(self, numel: int, element_size: int = 4):
        self._numel = numel
        self._element_size = element_size

    def clone(self) -> FakeTensor:
        return FakeTensor(self._numel, self._element_size)

    def copy_(self, other: FakeTensor) -> FakeTensor:
        assert self._numel == other._numel
        assert self._element_size == other._element_size
        return self

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class FakeEvent:

    def record(self) -> None:
        pass

    def elapsed_time(self, other: FakeEvent) -> float:
        del other
        return 0.1


class FakeNpu:

    @staticmethod
    def synchronize() -> None:
        pass

    @staticmethod
    def Event(*, enable_timing: bool) -> FakeEvent:
        assert enable_timing
        return FakeEvent()


class FakeTorch:
    npu = FakeNpu()


def make_case() -> Any:
    baseline_state = tuple(FakeTensor(size) for size in (10, 20, 30, 40))
    return BENCHMARK.DeviceCase(
        baseline_state=baseline_state,
        mutable_state=tuple(tensor.clone() for tensor in baseline_state),
        req_pool_entries=FakeTensor(2),
        query_index=FakeTensor(3),
        lookup_mask=FakeTensor(4),
        expected_slots=FakeTensor(5),
        expected_misses=FakeTensor(6),
    )


def test_clone_maintain_case_inputs_clones_every_operator_argument() -> None:
    case = make_case()

    first = BENCHMARK.clone_maintain_case_inputs(case)
    second = BENCHMARK.clone_maintain_case_inputs(case)

    assert first.baseline_state is case.baseline_state
    for source, first_tensor, second_tensor in zip(
        case.baseline_state,
        first.mutable_state,
        second.mutable_state,
        strict=True,
    ):
        assert first_tensor is not source
        assert second_tensor is not source
        assert first_tensor is not second_tensor
    assert first.req_pool_entries is not case.req_pool_entries
    assert first.req_pool_entries is not second.req_pool_entries
    assert first.expected_slots is not case.expected_slots
    assert first.expected_slots is not second.expected_slots


def test_maintain_input_bytes_counts_only_operator_inputs() -> None:
    case = make_case()

    assert BENCHMARK.maintain_input_bytes(case) == (
        10 + 20 + 30 + 40 + 2 + 5
    ) * 4


def test_benchmark_fresh_mode_uses_each_tensor_set_once() -> None:
    case = make_case()
    seen_argument_ids: list[tuple[int, ...]] = []

    def invoke(active_case: Any) -> None:
        seen_argument_ids.append(
            tuple(
                id(tensor)
                for tensor in (
                    *active_case.mutable_state,
                    active_case.req_pool_entries,
                    active_case.expected_slots,
                )
            )
        )

    samples = BENCHMARK.benchmark_case(
        FakeTorch(),
        case,
        invoke,
        warmup_iterations=2,
        iterations=3,
        fresh_maintain_tensors=True,
    )

    assert samples == [0.1, 0.1, 0.1]
    assert len(seen_argument_ids) == 5
    for argument_index in range(6):
        assert len({ids[argument_index] for ids in seen_argument_ids}) == 5
