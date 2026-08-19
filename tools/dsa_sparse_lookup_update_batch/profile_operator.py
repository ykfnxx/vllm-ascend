#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import math
from pathlib import Path

from common import (
    QUERY_WIDTH,
    clone_inputs,
    invoke,
    load_runtime,
    make_profile_inputs,
    restore_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--queries-per-request", type=int, default=4)
    parser.add_argument("--miss-rate", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dsa-sparse-batch-profile"),
    )
    args = parser.parse_args()
    if not 0 <= args.miss_rate <= 100:
        raise ValueError("miss-rate must be in [0, 100]")
    if args.warmup <= 0 or args.steps <= 0:
        raise ValueError("warmup and steps must be positive")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experimental_config = runtime.torch_npu.profiler._ExperimentalConfig(
        export_type=runtime.torch_npu.profiler.ExportType.Text,
    )
    with runtime.torch_npu.profiler.profile(
        activities=[runtime.torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=runtime.torch_npu.profiler.tensorboard_trace_handler(
            str(args.output_dir)
        ),
        record_shapes=True,
        experimental_config=experimental_config,
    ) as profiler:
        for _ in range(args.steps):
            restore_inputs(inputs, reference)
            invoke(runtime, inputs)
            profiler.step()
    runtime.torch.npu.synchronize()
    print(f"Profile written under: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
