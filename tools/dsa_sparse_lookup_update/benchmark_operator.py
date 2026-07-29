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

from common import (
    INDEX_CAPACITY,
    QUERY_COUNT,
    RESIDENT_SLOT_COUNT,
    TOOL_DIR,
    Runtime,
    invoke,
    load_runtime,
    make_profile_inputs,
    validate_requests,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one fused DSA Sparse metadata invocation "
            "with 8K resident entries per concurrent request."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--scenario",
        choices=("hit", "churn", "both"),
        default="both",
    )
    miss_group = parser.add_mutually_exclusive_group()
    miss_group.add_argument("--miss-rate", type=float)
    miss_group.add_argument("--miss-count", type=int)
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


def _query_groups(
    runtime: Runtime,
    *,
    concurrency: int,
    miss_count: int,
) -> list:
    torch = runtime.torch
    hit_count = QUERY_COUNT - miss_count
    hit_tokens = torch.arange(
        hit_count,
        dtype=torch.int32,
        device=runtime.device,
    )
    if miss_count == 0:
        return [
            hit_tokens.expand(concurrency, -1).contiguous()
        ]

    replaceable = RESIDENT_SLOT_COUNT - hit_count
    group_count = math.ceil(replaceable / miss_count) + 1
    last_token = (
        RESIDENT_SLOT_COUNT + group_count * miss_count
    )
    if last_token > INDEX_CAPACITY:
        raise ValueError(
            "controlled-miss token groups exceed index capacity"
        )
    groups = []
    for group in range(group_count):
        misses = torch.arange(
            RESIDENT_SLOT_COUNT + group * miss_count,
            RESIDENT_SLOT_COUNT + (group + 1) * miss_count,
            dtype=torch.int32,
            device=runtime.device,
        )
        row = torch.cat((hit_tokens, misses))
        groups.append(
            row.expand(concurrency, -1).contiguous()
        )
    return groups


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[
        max(0, math.ceil(percentile * len(ordered)) - 1)
    ]


def _run_benchmark(
    runtime: Runtime,
    *,
    concurrency: int,
    miss_count: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    torch = runtime.torch
    inputs = make_profile_inputs(
        runtime, requests=concurrency
    )
    groups = _query_groups(
        runtime,
        concurrency=concurrency,
        miss_count=miss_count,
    )

    inputs.query_index.copy_(groups[0])
    _, validation_misses = invoke(runtime, inputs)
    torch.npu.synchronize()
    actual_misses = int(validation_misses.sum().item())
    expected_misses = concurrency * miss_count
    if actual_misses != expected_misses:
        raise AssertionError(
            f"controlled miss validation got {actual_misses}, "
            f"expected {expected_misses}"
        )

    group_index = 1
    for _ in range(warmup):
        inputs.query_index.copy_(
            groups[group_index % len(groups)]
        )
        invoke(runtime, inputs)
        group_index += 1
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
        inputs.query_index.copy_(
            groups[group_index % len(groups)]
        )
        group_index += 1
        starts[iteration].record()
        invoke(runtime, inputs)
        ends[iteration].record()
    torch.npu.synchronize()
    samples_us = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends)
    ]
    return {
        "miss_count_per_request": miss_count,
        "effective_miss_rate_percent": (
            100.0 * miss_count / QUERY_COUNT
        ),
        "samples": len(samples_us),
        "min_us": min(samples_us),
        "median_us": statistics.median(samples_us),
        "p95_us": _percentile(samples_us, 0.95),
        "max_us": max(samples_us),
        "mean_us": statistics.fmean(samples_us),
    }


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
    runtime = load_runtime(
        device=args.device,
        install_root=args.install_root,
    )
    with runtime.torch.inference_mode():
        results = {
            name: _run_benchmark(
                runtime,
                concurrency=args.concurrency,
                miss_count=miss_count,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            for name, miss_count in _workloads(args)
        }
    manifest = {
        "operator": "dsa_sparse_lookup_update",
        "device": args.device,
        "device_name": _device_name(runtime),
        "git_head": _git_head(),
        "torch_version": runtime.torch.__version__,
        "torch_npu_version": runtime.torch_npu.__version__,
        "install_root": (
            str(runtime.install_root)
            if runtime.install_root is not None
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
