#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from common import (
    QUERY_WIDTH,
    SLOT_COUNT,
    clone_inputs,
    invoke,
    load_runtime,
    make_profile_inputs,
    restore_inputs,
)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Time only dsa_sparse_lookup_update_batch. State restore is "
            "issued before the start event and excluded from event duration."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--queries-per-request", type=int, default=4)
    parser.add_argument("--miss-rate", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not 0 <= args.miss_rate <= 100:
        raise ValueError("miss-rate must be in [0, 100]")
    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    miss_count = math.floor(QUERY_WIDTH * args.miss_rate / 100 + 0.5)

    runtime = load_runtime(
        device=args.device,
        install_root=args.install_root,
    )
    reference = make_profile_inputs(
        runtime,
        requests=args.concurrency,
        queries_per_request=args.queries_per_request,
        miss_count_per_query=miss_count,
    )
    inputs = clone_inputs(reference)
    for _ in range(args.warmup):
        restore_inputs(inputs, reference)
        invoke(runtime, inputs)
    runtime.torch.npu.synchronize()

    starts = [
        runtime.torch.npu.Event(enable_timing=True)
        for _ in range(args.iterations)
    ]
    ends = [
        runtime.torch.npu.Event(enable_timing=True)
        for _ in range(args.iterations)
    ]
    slot_out = None
    miss_out = None
    for iteration in range(args.iterations):
        restore_inputs(inputs, reference)
        starts[iteration].record()
        slot_out, miss_out = invoke(runtime, inputs)
        ends[iteration].record()
    runtime.torch.npu.synchronize()
    assert slot_out is not None and miss_out is not None
    samples_us = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends)
    ]
    result = {
        "operator": "dsa_sparse_lookup_update_batch",
        "timed_region": "operator_only",
        "state_restore_excluded": True,
        "resident_slots_per_request": 8192,
        "concurrency": args.concurrency,
        "queries_per_request": args.queries_per_request,
        "packed_query_rows": (
            args.concurrency * args.queries_per_request
        ),
        "miss_count_per_query": miss_count,
        "requested_miss_rate_percent": args.miss_rate,
        "actual_miss_count": int(miss_out.sum().item()),
        "fallback_count": int(slot_out.eq(SLOT_COUNT).sum().item()),
        "samples": len(samples_us),
        "min_us": min(samples_us),
        "median_us": statistics.median(samples_us),
        "p95_us": _percentile(samples_us, 0.95),
        "max_us": max(samples_us),
        "mean_us": statistics.fmean(samples_us),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
