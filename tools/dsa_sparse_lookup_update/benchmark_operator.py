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
    TOOL_DIR,
    OperatorInputs,
    Runtime,
    invoke,
    load_runtime,
    make_profile_inputs,
)

RESIDENT_CACHE_SLOTS = 8192
QUERY_LANES = 1
DEFAULT_TOPK = 2048


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one Ascend 950 dsa_sparse_lookup_update invocation with "
            "an 8K resident cache per concurrent request."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--scenario",
        choices=("hit", "churn", "both"),
        default="both",
        help=(
            "hit measures resident lookup/LRU maintenance; churn measures "
            "full-cache miss allocation and replacement. Overridden by "
            "--miss-rate or --miss-count."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    miss_group = parser.add_mutually_exclusive_group()
    miss_group.add_argument(
        "--miss-rate",
        type=float,
        help=(
            "Requested miss percentage in each request's Top-K. The nearest "
            "integer miss count is used."
        ),
    )
    miss_group.add_argument(
        "--miss-count",
        type=int,
        help="Exact number of misses in each request's Top-K.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("concurrency", "max_model_len", "topk", "warmup", "iterations"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(
                f"{name.replace('_', '-')} must be positive, got {value}."
            )
    if args.topk > RESIDENT_CACHE_SLOTS:
        raise ValueError(
            f"topk must not exceed the 8K resident cache, got {args.topk}."
        )
    if args.miss_rate is not None and not 0.0 <= args.miss_rate <= 100.0:
        raise ValueError(
            f"miss-rate must be in [0, 100], got {args.miss_rate}."
        )
    if args.miss_count is not None and not 0 <= args.miss_count <= args.topk:
        raise ValueError(
            f"miss-count must be in [0, {args.topk}], got {args.miss_count}."
        )
    if args.max_model_len <= RESIDENT_CACHE_SLOTS:
        raise ValueError(
            "max-model-len must leave room for the 8K resident set and the "
            f"current query position; need at least {RESIDENT_CACHE_SLOTS + 1}, "
            f"got {args.max_model_len}."
        )


def _requested_miss_count(args: argparse.Namespace) -> int | None:
    if args.miss_count is not None:
        return args.miss_count
    if args.miss_rate is not None:
        return math.floor(args.topk * args.miss_rate / 100.0 + 0.5)
    return None


def _workloads(args: argparse.Namespace) -> list[tuple[str, int]]:
    requested = _requested_miss_count(args)
    if requested is not None:
        return [("custom", requested)]
    if args.scenario == "hit":
        return [("hit", 0)]
    if args.scenario == "churn":
        return [("churn", args.topk)]
    return [("hit", 0), ("churn", args.topk)]


def _populate_resident_cache(
    runtime: Runtime,
    inputs: OperatorInputs,
    *,
    concurrency: int,
) -> None:
    torch = runtime.torch
    slots = torch.arange(
        RESIDENT_CACHE_SLOTS,
        dtype=torch.int32,
        device=runtime.device,
    )
    resident = slots.expand(concurrency, -1)
    inputs.token_to_hot[:, :RESIDENT_CACHE_SLOTS].copy_(resident)
    inputs.hot_to_token.copy_(resident)
    inputs.lru_slots.copy_(resident)
    inputs.state_seat_epoch.zero_()
    inputs.row_seat_epoch.zero_()


def _make_topk_groups(
    runtime: Runtime,
    *,
    concurrency: int,
    topk: int,
    miss_count: int,
    max_model_len: int,
) -> list[Any]:
    torch = runtime.torch
    hit_count = topk - miss_count
    hit_tokens = torch.arange(
        hit_count,
        dtype=torch.int32,
        device=runtime.device,
    )
    if miss_count == 0:
        return [
            hit_tokens.expand(concurrency, -1).contiguous()
        ]

    replaceable_slots = RESIDENT_CACHE_SLOTS - hit_count
    group_count = math.ceil(replaceable_slots / miss_count) + 1
    first_miss_token = RESIDENT_CACHE_SLOTS
    required_token_count = first_miss_token + group_count * miss_count
    if max_model_len <= required_token_count:
        raise ValueError(
            "max-model-len is too small for the requested controlled-miss "
            f"workload; need at least {required_token_count + 1}, got "
            f"{max_model_len}."
        )

    groups = []
    for group in range(group_count):
        miss_tokens = torch.arange(
            first_miss_token + group * miss_count,
            first_miss_token + (group + 1) * miss_count,
            dtype=torch.int32,
            device=runtime.device,
        )
        row_topk = torch.cat((hit_tokens, miss_tokens))
        groups.append(row_topk.expand(concurrency, -1).contiguous())
    return groups


def _percentile(sorted_values: list[float], percentile: float) -> float:
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[index]


def _summarize_us(
    samples_us: list[float],
    *,
    concurrency: int,
    topk: int,
    miss_count: int,
) -> dict[str, float | int]:
    sorted_samples = sorted(samples_us)
    mean_us = statistics.fmean(sorted_samples)
    return {
        "count": len(sorted_samples),
        "min_us": sorted_samples[0],
        "median_us": statistics.median(sorted_samples),
        "p95_us": _percentile(sorted_samples, 0.95),
        "max_us": sorted_samples[-1],
        "mean_us": mean_us,
        "batched_requests_per_second": concurrency * 1_000_000.0 / mean_us,
        "topk_entries_per_second": (
            concurrency * QUERY_LANES * topk * 1_000_000.0 / mean_us
        ),
        "miss_count_per_request": miss_count,
        "miss_count_per_invocation": concurrency * miss_count,
        "effective_miss_rate_percent": miss_count * 100.0 / topk,
    }


def _group_for_iteration(
    groups: list[Any],
    iteration: int,
) -> Any:
    return groups[iteration % len(groups)]


def _validate_device_miss_count(
    runtime: Runtime,
    inputs: OperatorInputs,
    *,
    topk_group: Any,
    expected_total_misses: int,
) -> None:
    inputs.topk_positions = topk_group
    invoke(runtime, inputs)
    runtime.torch.npu.synchronize()
    actual_total_misses = int(inputs.miss_mask.sum().item())
    if actual_total_misses != expected_total_misses:
        raise RuntimeError(
            "Controlled-miss workload validation failed: "
            f"expected {expected_total_misses} misses, got "
            f"{actual_total_misses}."
        )


def _run_benchmark(
    runtime: Runtime,
    *,
    concurrency: int,
    max_model_len: int,
    topk: int,
    miss_count: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    torch = runtime.torch
    inputs = make_profile_inputs(
        runtime,
        seats=concurrency,
        rows=concurrency,
        max_model_len=max_model_len,
        slots=RESIDENT_CACHE_SLOTS,
        lanes=QUERY_LANES,
        topk=topk,
    )
    _populate_resident_cache(
        runtime,
        inputs,
        concurrency=concurrency,
    )
    groups = _make_topk_groups(
        runtime,
        concurrency=concurrency,
        topk=topk,
        miss_count=miss_count,
        max_model_len=max_model_len,
    )
    torch.npu.synchronize()

    _validate_device_miss_count(
        runtime,
        inputs,
        topk_group=groups[0],
        expected_total_misses=concurrency * miss_count,
    )

    iteration_offset = 1
    for iteration in range(warmup):
        inputs.topk_positions = _group_for_iteration(
            groups,
            iteration=iteration_offset + iteration,
        )
        invoke(runtime, inputs)
    torch.npu.synchronize()

    start_events = [
        torch.npu.Event(enable_timing=True) for _ in range(iterations)
    ]
    end_events = [
        torch.npu.Event(enable_timing=True) for _ in range(iterations)
    ]
    for iteration in range(iterations):
        inputs.topk_positions = _group_for_iteration(
            groups,
            iteration=iteration_offset + warmup + iteration,
        )
        start_events[iteration].record()
        invoke(runtime, inputs)
        end_events[iteration].record()
    torch.npu.synchronize()

    samples_us = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(start_events, end_events)
    ]
    return _summarize_us(
        samples_us,
        concurrency=concurrency,
        topk=topk,
        miss_count=miss_count,
    )


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=TOOL_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _device_name(runtime: Runtime) -> str:
    try:
        return str(runtime.torch.npu.get_device_name(runtime.device))
    except Exception:
        return "unknown"


def _output_path(requested: Path | None) -> Path:
    if requested is not None:
        output = requested.expanduser().resolve()
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output = TOOL_DIR / "benchmarks" / f"{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    return output


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    runtime = load_runtime(
        device=args.device,
        install_root=args.install_root,
    )
    workloads = _workloads(args)

    with runtime.torch.inference_mode():
        results = {
            name: _run_benchmark(
                runtime,
                concurrency=args.concurrency,
                max_model_len=args.max_model_len,
                topk=args.topk,
                miss_count=miss_count,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            for name, miss_count in workloads
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
            "cache_seats": args.concurrency,
            "resident_cache_slots_per_request": RESIDENT_CACHE_SLOTS,
            "total_resident_cache_slots": (
                args.concurrency * RESIDENT_CACHE_SLOTS
            ),
            "max_model_len": args.max_model_len,
            "query_lanes": QUERY_LANES,
            "topk": args.topk,
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
