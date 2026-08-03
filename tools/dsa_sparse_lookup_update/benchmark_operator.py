#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    INVALID_INDEX,
    LOOKUP_OPERATOR,
    MAX_SEED,
    MAINTAIN_OPERATOR,
    QUERY_COUNT,
    RESIDENT_SLOT_COUNT,
    SIMT_OPERATOR,
    TOOL_DIR,
    MaintainInputs,
    OperatorInputs,
    Runtime,
    clone_maintain_inputs,
    clone_operator_inputs,
    invoke,
    invoke_maintain,
    load_runtime,
    make_maintain_profile_inputs,
    make_profile_inputs,
    restore_maintain_inputs,
    restore_operator_inputs,
    validate_requests,
)

OPERATOR_SELECTIONS = {
    "simt": (SIMT_OPERATOR,),
    "lookup": (LOOKUP_OPERATOR,),
    "maintain": (MAINTAIN_OPERATOR,),
    "legacy": (LOOKUP_OPERATOR, MAINTAIN_OPERATOR),
    "all": (SIMT_OPERATOR, LOOKUP_OPERATOR, MAINTAIN_OPERATOR),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the SIMT fused operator, ASU lookup, and AICPU "
            "maintain independently with 8K resident entries per request."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument(
        "--operator",
        choices=tuple(OPERATOR_SELECTIONS),
        default="all",
        help=(
            "legacy and all run multiple independent benchmark phases; "
            "they never time an operator pair."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--scenario",
        choices=("hit", "churn", "both"),
        default="both",
    )
    miss_group = parser.add_mutually_exclusive_group()
    miss_group.add_argument("--miss-rate", type=float)
    miss_group.add_argument("--miss-count", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    validate_requests(args.concurrency)
    for name in ("warmup", "iterations"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        args.miss_rate is not None
        and not 0.0 <= args.miss_rate <= 100.0
    ):
        raise ValueError("miss-rate must be in [0, 100]")
    if (
        args.miss_count is not None
        and not 0 <= args.miss_count <= QUERY_COUNT
    ):
        raise ValueError(
            f"miss-count must be in [0, {QUERY_COUNT}]"
        )
    if not 0 <= args.seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}]")


def _requested_misses(args: argparse.Namespace) -> int | None:
    if args.miss_count is not None:
        return args.miss_count
    if args.miss_rate is not None:
        return math.floor(
            QUERY_COUNT * args.miss_rate / 100.0 + 0.5
        )
    return None


def _workloads(args: argparse.Namespace) -> list[tuple[str, int]]:
    requested = _requested_misses(args)
    if requested is not None:
        return [("custom", requested)]
    if args.scenario == "hit":
        return [("hit", 0)]
    if args.scenario == "churn":
        return [("churn", QUERY_COUNT)]
    return [("hit", 0), ("churn", QUERY_COUNT)]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[
        max(0, math.ceil(percentile * len(ordered)) - 1)
    ]


def _summarize(
    samples_us: list[float],
    *,
    miss_count: int,
) -> dict[str, float | int | bool]:
    return {
        "miss_count_per_request": miss_count,
        "effective_miss_rate_percent": (
            100.0 * miss_count / QUERY_COUNT
        ),
        "state_reset_excluded": True,
        "samples": len(samples_us),
        "min_us": min(samples_us),
        "median_us": statistics.median(samples_us),
        "p95_us": _percentile(samples_us, 0.95),
        "max_us": max(samples_us),
        "mean_us": statistics.fmean(samples_us),
    }


def _validate_lookup_result(
    runtime: Runtime,
    inputs: OperatorInputs,
    miss_out: Any,
    *,
    concurrency: int,
    miss_count: int,
    fused_maintenance: bool,
) -> None:
    actual_misses = int(miss_out.sum().item())
    expected_misses = concurrency * miss_count
    if actual_misses != expected_misses:
        raise AssertionError(
            f"{runtime.operator_name} produced {actual_misses} "
            f"misses, expected {expected_misses}"
        )
    actual_free_head = int(inputs.free_head[:, 0].sum().item())
    expected_free_head = (
        0 if fused_maintenance else expected_misses
    )
    if actual_free_head != expected_free_head:
        raise AssertionError(
            f"{runtime.operator_name} left free_head sum "
            f"{actual_free_head}, expected {expected_free_head}"
        )
    occupied_slots = int(
        inputs.slot_to_index.ne(INVALID_INDEX).sum().item()
    )
    occupied_per_request = (
        RESIDENT_SLOT_COUNT
        if fused_maintenance
        else RESIDENT_SLOT_COUNT + miss_count
    )
    expected_occupied = concurrency * occupied_per_request
    if occupied_slots != expected_occupied:
        raise AssertionError(
            f"{runtime.operator_name} left {occupied_slots} "
            f"occupied slots, expected {expected_occupied}"
        )


def _run_lookup_like_benchmark(
    runtime: Runtime,
    *,
    concurrency: int,
    miss_count: int,
    seed: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | bool]:
    torch = runtime.torch
    reference = make_profile_inputs(
        runtime,
        requests=concurrency,
        miss_count=miss_count,
        seed=seed,
    )
    inputs = clone_operator_inputs(reference)
    fused_maintenance = runtime.operator_name == SIMT_OPERATOR

    restore_operator_inputs(inputs, reference)
    _, validation_misses = invoke(runtime, inputs)
    torch.npu.synchronize()
    _validate_lookup_result(
        runtime,
        inputs,
        validation_misses,
        concurrency=concurrency,
        miss_count=miss_count,
        fused_maintenance=fused_maintenance,
    )

    for _ in range(warmup):
        restore_operator_inputs(inputs, reference)
        invoke(runtime, inputs)
    torch.npu.synchronize()

    starts = [
        torch.npu.Event(enable_timing=True)
        for _ in range(iterations)
    ]
    ends = [
        torch.npu.Event(enable_timing=True)
        for _ in range(iterations)
    ]
    final_misses = validation_misses
    for iteration in range(iterations):
        restore_operator_inputs(inputs, reference)
        starts[iteration].record()
        _, final_misses = invoke(runtime, inputs)
        ends[iteration].record()
    torch.npu.synchronize()
    _validate_lookup_result(
        runtime,
        inputs,
        final_misses,
        concurrency=concurrency,
        miss_count=miss_count,
        fused_maintenance=fused_maintenance,
    )
    samples_us = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends)
    ]
    result = _summarize(
        samples_us,
        miss_count=miss_count,
    )
    result["validated_misses_per_invocation"] = (
        concurrency * miss_count
    )
    return result


def _validate_maintain_result(
    inputs: MaintainInputs,
    reference: MaintainInputs,
    *,
    concurrency: int,
) -> None:
    actual_free_head = int(inputs.free_head[:, 0].sum().item())
    if actual_free_head != 0:
        raise AssertionError(
            "asu_hbm_index_maintain_aicpu left free_head sum "
            f"{actual_free_head}, expected 0"
        )
    occupied_slots = int(
        inputs.slot_to_index.ne(INVALID_INDEX).sum().item()
    )
    expected_occupied = concurrency * RESIDENT_SLOT_COUNT
    if occupied_slots != expected_occupied:
        raise AssertionError(
            "asu_hbm_index_maintain_aicpu left "
            f"{occupied_slots} occupied slots, "
            f"expected {expected_occupied}"
        )
    protected_slots = reference.last_query_slots.long()
    expected_protected_tokens = reference.slot_to_index.gather(
        1,
        protected_slots,
    )
    actual_protected_tokens = inputs.slot_to_index.gather(
        1,
        protected_slots,
    )
    if not bool(
        actual_protected_tokens.eq(
            expected_protected_tokens
        ).all().item()
    ):
        raise AssertionError(
            "asu_hbm_index_maintain_aicpu evicted a protected slot"
        )


def _run_maintain_benchmark(
    runtime: Runtime,
    *,
    concurrency: int,
    miss_count: int,
    seed: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | bool]:
    torch = runtime.torch
    reference = make_maintain_profile_inputs(
        runtime,
        requests=concurrency,
        miss_count=miss_count,
        seed=seed,
    )
    inputs = clone_maintain_inputs(reference)
    seed_cursor = seed

    restore_maintain_inputs(inputs, reference)
    invoke_maintain(runtime, inputs, seed=seed_cursor)
    seed_cursor = (seed_cursor + 1) & MAX_SEED
    torch.npu.synchronize()
    _validate_maintain_result(
        inputs,
        reference,
        concurrency=concurrency,
    )

    for _ in range(warmup):
        restore_maintain_inputs(inputs, reference)
        invoke_maintain(runtime, inputs, seed=seed_cursor)
        seed_cursor = (seed_cursor + 1) & MAX_SEED
    torch.npu.synchronize()

    starts = [
        torch.npu.Event(enable_timing=True)
        for _ in range(iterations)
    ]
    ends = [
        torch.npu.Event(enable_timing=True)
        for _ in range(iterations)
    ]
    for iteration in range(iterations):
        restore_maintain_inputs(inputs, reference)
        starts[iteration].record()
        invoke_maintain(
            runtime,
            inputs,
            seed=seed_cursor,
        )
        ends[iteration].record()
        seed_cursor = (seed_cursor + 1) & MAX_SEED
    torch.npu.synchronize()
    _validate_maintain_result(
        inputs,
        reference,
        concurrency=concurrency,
    )
    samples_us = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends)
    ]
    result = _summarize(
        samples_us,
        miss_count=miss_count,
    )
    result["pending_evictions_per_invocation"] = (
        concurrency * miss_count
    )
    result["seed_start"] = seed
    result["seed_next"] = seed_cursor
    return result


def _run_operator_benchmark(
    runtime: Runtime,
    *,
    concurrency: int,
    miss_count: int,
    seed: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | bool]:
    if runtime.operator_name == MAINTAIN_OPERATOR:
        return _run_maintain_benchmark(
            runtime,
            concurrency=concurrency,
            miss_count=miss_count,
            seed=seed,
            warmup=warmup,
            iterations=iterations,
        )
    return _run_lookup_like_benchmark(
        runtime,
        concurrency=concurrency,
        miss_count=miss_count,
        seed=seed,
        warmup=warmup,
        iterations=iterations,
    )


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=TOOL_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        result.stdout.strip()
        if result.returncode == 0
        else None
    )


def _device_name(runtime: Runtime) -> str:
    try:
        return str(
            runtime.torch.npu.get_device_name(runtime.device)
        )
    except Exception:
        return "unknown"


def _output_path(requested: Path | None) -> Path:
    if requested is not None:
        output = requested.expanduser().resolve()
    else:
        timestamp = (
            datetime.now()
            .astimezone()
            .strftime("%Y%m%d-%H%M%S")
        )
        output = (
            TOOL_DIR / "benchmarks" / f"{timestamp}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}"
        )
    return output


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    operator_names = OPERATOR_SELECTIONS[args.operator]
    runtimes = {
        operator_name: load_runtime(
            device=args.device,
            install_root=args.install_root,
            operator_name=operator_name,
        )
        for operator_name in operator_names
    }
    first_runtime = runtimes[operator_names[0]]
    with first_runtime.torch.inference_mode():
        results = {
            operator_name: {
                workload_name: _run_operator_benchmark(
                    runtimes[operator_name],
                    concurrency=args.concurrency,
                    miss_count=miss_count,
                    seed=args.seed,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                for workload_name, miss_count in _workloads(args)
            }
            for operator_name in operator_names
        }
    manifest = {
        "selection": args.operator,
        "operators": list(operator_names),
        "measurement_scope": (
            "one selected operator per NPU Event interval"
        ),
        "device": args.device,
        "device_name": _device_name(first_runtime),
        "git_head": _git_head(),
        "torch_version": first_runtime.torch.__version__,
        "torch_npu_version": first_runtime.torch_npu.__version__,
        "install_root": (
            str(first_runtime.install_root)
            if first_runtime.install_root is not None
            else None
        ),
        "workload": {
            "layer_count": 1,
            "concurrency": args.concurrency,
            "resident_entries_per_request": (
                RESIDENT_SLOT_COUNT
            ),
            "query_width": QUERY_COUNT,
            "requested_miss_rate_percent": args.miss_rate,
            "requested_miss_count": args.miss_count,
            "query_pattern": "seeded-random-resident-and-query",
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "results": results,
    }
    output = _output_path(args.output)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
