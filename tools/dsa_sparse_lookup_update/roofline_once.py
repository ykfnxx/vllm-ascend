#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import math
from pathlib import Path

from common import (
    MAX_SEED,
    QUERY_COUNT,
    SIMT_OPERATOR,
    invoke,
    load_runtime,
    make_profile_inputs,
    validate_requests,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one DsaSparseLookupUpdate invocation for "
            "msOpProf Roofline collection."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--requests", type=int, default=32)
    miss_group = parser.add_mutually_exclusive_group()
    miss_group.add_argument("--miss-rate", type=float)
    miss_group.add_argument("--miss-count", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _requested_misses(args: argparse.Namespace) -> int:
    if args.miss_count is not None:
        return args.miss_count
    miss_rate = 10.0 if args.miss_rate is None else args.miss_rate
    return math.floor(QUERY_COUNT * miss_rate / 100.0 + 0.5)


def _validate_args(args: argparse.Namespace) -> None:
    validate_requests(args.requests)
    if args.miss_rate is not None and not 0.0 <= args.miss_rate <= 100.0:
        raise ValueError("miss-rate must be in [0, 100]")
    if (
        args.miss_count is not None
        and not 0 <= args.miss_count <= QUERY_COUNT
    ):
        raise ValueError(f"miss-count must be in [0, {QUERY_COUNT}]")
    if not 0 <= args.seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}]")


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    miss_count = _requested_misses(args)
    runtime = load_runtime(
        device=args.device,
        install_root=args.install_root,
        operator_name=SIMT_OPERATOR,
    )
    inputs = make_profile_inputs(
        runtime,
        requests=args.requests,
        miss_count=miss_count,
        seed=args.seed,
    )

    # Keep workload construction outside the profiled kernel interval. The
    # process contains exactly one DsaSparseLookupUpdate launch.
    runtime.torch.npu.synchronize()
    slot_out, miss_out = invoke(runtime, inputs)
    runtime.torch.npu.synchronize()

    if not args.quiet:
        print(
            "DsaSparseLookupUpdate one-shot workload: "
            f"requests={args.requests}, misses_per_request={miss_count}, "
            f"seed={args.seed}, slot_out_shape={tuple(slot_out.shape)}, "
            f"miss_out_shape={tuple(miss_out.shape)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
