# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOL_DIR = ROOT / "tools" / "dsa_sparse_lookup_update"
SCRIPT_PATH = TOOL_DIR / "profile_operator_matrix.py"
sys.path.insert(0, str(TOOL_DIR))
SPEC = importlib.util.spec_from_file_location(
    "dsa_sparse_profile_matrix",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
PROFILE_MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE_MATRIX
SPEC.loader.exec_module(PROFILE_MATRIX)


def test_default_plan_covers_fixed_optimization_workloads() -> None:
    args = PROFILE_MATRIX._parse_args([])

    workloads = PROFILE_MATRIX._workloads(args)
    metrics = PROFILE_MATRIX._metric_profiles(args.metrics)

    assert [
        (workload.requests, workload.miss_count)
        for workload in workloads
    ] == [
        (32, 0),
        (32, 1),
        (32, 205),
        (32, 2048),
    ]
    assert [metric.name for metric in metrics] == [
        "pipe-utilization",
        "memory",
        "resource-conflict",
        "l2-cache",
    ]
    assert args.seed == 0


def test_multiple_request_counts_and_miss_rates_form_matrix() -> None:
    args = PROFILE_MATRIX._parse_args(
        [
            "--requests",
            "1",
            "32",
            "--miss-rates",
            "0",
            "10",
            "100",
        ]
    )

    workloads = PROFILE_MATRIX._workloads(args)

    assert [
        (workload.requests, workload.miss_count)
        for workload in workloads
    ] == [
        (1, 0),
        (1, 205),
        (1, 2048),
        (32, 0),
        (32, 205),
        (32, 2048),
    ]


def test_duplicate_inputs_are_deduplicated_in_stable_order() -> None:
    args = PROFILE_MATRIX._parse_args(
        [
            "--requests",
            "32",
            "1",
            "32",
            "--miss-counts",
            "205",
            "0",
            "205",
            "--metrics",
            "memory",
            "pipe-utilization",
            "memory",
        ]
    )

    workloads = PROFILE_MATRIX._workloads(args)
    metrics = PROFILE_MATRIX._metric_profiles(args.metrics)

    assert [
        (workload.requests, workload.miss_count)
        for workload in workloads
    ] == [
        (32, 205),
        (32, 0),
        (1, 205),
        (1, 0),
    ]
    assert [metric.name for metric in metrics] == [
        "memory",
        "pipe-utilization",
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (["--requests", "0"], "requests"),
        (["--miss-counts", "-1"], "miss count"),
        (["--miss-counts", "2049"], "miss count"),
        (["--miss-rates", "-0.1"], "miss rate"),
        (["--miss-rates", "100.1"], "miss rate"),
    ),
)
def test_invalid_workload_values_are_rejected(
    arguments: list[str],
    message: str,
) -> None:
    args = PROFILE_MATRIX._parse_args(arguments)

    with pytest.raises(ValueError, match=message):
        PROFILE_MATRIX._workloads(args)


def test_workload_directory_name_is_stable() -> None:
    workload = PROFILE_MATRIX.Workload(
        requests=32,
        miss_count=205,
    )

    assert workload.name == "req-0032_miss-0205"


def test_profile_matrix_accepts_reproducible_workload_seed() -> None:
    args = PROFILE_MATRIX._parse_args(["--seed", "1234"])

    PROFILE_MATRIX._validate_iterations(args)

    assert args.seed == 1234


@pytest.mark.parametrize(
    "mode",
    ("steady", "step-random", "cache-thrash"),
)
def test_profile_matrix_accepts_workload_modes(mode: str) -> None:
    args = PROFILE_MATRIX._parse_args(["--workload", mode])

    assert args.workload_mode == mode


def test_profile_matrix_accepts_workload_mode_alias() -> None:
    args = PROFILE_MATRIX._parse_args(
        ["--workload-mode", "cache-thrash"]
    )

    assert args.workload_mode == "cache-thrash"


@pytest.mark.parametrize("seed", ("-1", "2147483648"))
def test_profile_matrix_rejects_invalid_seed(seed: str) -> None:
    args = PROFILE_MATRIX._parse_args(["--seed", seed])

    with pytest.raises(ValueError, match="seed"):
        PROFILE_MATRIX._validate_iterations(args)
