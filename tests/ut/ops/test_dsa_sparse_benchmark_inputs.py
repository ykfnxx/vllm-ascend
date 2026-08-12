# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import importlib.util
import subprocess
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
ROOFLINE_SCRIPT = (
    ROOT
    / "tools"
    / "dsa_sparse_lookup_update"
    / "profile_roofline.sh"
)
ROOFLINE_RUNNER = (
    ROOT
    / "tools"
    / "dsa_sparse_lookup_update"
    / "standalone"
    / "runner"
    / "dsa_sparse_lookup_update_runner.cpp"
)
STANDALONE_ROOT = (
    ROOT
    / "tools"
    / "dsa_sparse_lookup_update"
    / "standalone"
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


def test_roofline_profiles_stateful_operator_with_application_replay() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOFLINE_SCRIPT),
            "--tool",
            "msopprof",
            "--warm-up",
            "3",
            "--miss-count",
            "205",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output_lines = result.stdout.splitlines()
    prewarm_command = next(
        line for line in output_lines if line.startswith("Prewarm command:")
    )
    roofline_command = next(
        line for line in output_lines if line.startswith("Roofline command:")
    )

    assert "dsa_sparse_lookup_update_runner" in prewarm_command
    assert "# repeat=3" in prewarm_command
    assert "--replay-mode=application" in roofline_command
    assert "--warm-up=0" in roofline_command
    assert "dsa_sparse_lookup_update_runner" in roofline_command
    assert "roofline_once.py" not in roofline_command
    assert "benchmark_operator.py" not in roofline_command
    assert "--warmup" not in roofline_command
    assert "--iterations" not in roofline_command


def test_roofline_accepts_kernel_replay() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOFLINE_SCRIPT),
            "--tool",
            "msopprof",
            "--replay-mode",
            "kernel",
            "--profiler-warm-up",
            "0",
            "--miss-rate",
            "0",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    roofline_command = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("Roofline command:")
    )

    assert "--replay-mode=kernel" in roofline_command
    assert "--warm-up=0" in roofline_command


def test_roofline_runner_contains_one_target_invocation() -> None:
    runner_source = ROOFLINE_RUNNER.read_text(encoding="utf-8")
    script_source = ROOFLINE_SCRIPT.read_text(encoding="utf-8")

    assert runner_source.count("aclnnDsaSparseLookupUpdate(") == 1
    assert "benchmark_operator.py" not in script_source
    assert "roofline_once.py" not in script_source


def test_standalone_kernel_sources_match_production_sources() -> None:
    production_root = (
        ROOT
        / "csrc"
        / "attention"
        / "dsa_sparse_lookup_update"
    )
    relative_sources = (
        "op_host/dsa_sparse_lookup_update_def.cpp",
        "op_host/dsa_sparse_lookup_update_infershape.cpp",
        "op_host/dsa_sparse_lookup_update_tiling.cpp",
        "op_host/dsa_sparse_lookup_update_tiling.h",
        "op_host/op_api/aclnn_dsa_sparse_lookup_update.cpp",
        "op_host/op_api/aclnn_dsa_sparse_lookup_update.h",
        "op_kernel/dsa_sparse_lookup_update.cpp",
        "op_kernel/dsa_sparse_lookup_update_common.h",
        "op_kernel/arch35/dsa_sparse_lookup_update_simt.h",
    )

    for relative_source in relative_sources:
        assert (STANDALONE_ROOT / relative_source).read_bytes() == (
            production_root / relative_source
        ).read_bytes()


def test_standalone_kernel_build_keeps_profile_information() -> None:
    kernel_cmake = (
        STANDALONE_ROOT / "op_kernel" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")
    root_cmake = (STANDALONE_ROOT / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "npu_op_kernel_options(" in kernel_cmake
    assert "-g" in kernel_cmake
    assert "-O0" not in kernel_cmake
    assert "npu_op_package(" in root_cmake
    assert "vllm" not in root_cmake.lower()
