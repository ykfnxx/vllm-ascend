#!/usr/bin/env python3
"""Benchmark the ASU HBM lookup and maintain custom operators on NPU."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any


INDEX_SIZE = 128 * 1024
SLOT_COUNT = 10 * 1024
FREE_SLOT_COUNT = 2 * 1024
QUERY_COUNT = 2 * 1024
FREE_HEAD_STRIDE = 16
RESIDENT_COUNT = SLOT_COUNT - FREE_SLOT_COUNT
MISS_TOKEN_BASE = INDEX_SIZE // 2
NOT_FOUND = -1


@dataclass
class HostCase:
    mutable_state: tuple[Any, Any, Any, Any]
    req_pool_entries: Any
    query_index: Any
    expected_slots: Any
    expected_misses: Any


@dataclass
class DeviceCase:
    baseline_state: tuple[Any, Any, Any, Any]
    mutable_state: tuple[Any, Any, Any, Any]
    req_pool_entries: Any
    query_index: Any
    expected_slots: Any
    expected_misses: Any

    def reset(self) -> None:
        for tensor, baseline in zip(
            self.mutable_state, self.baseline_state, strict=True
        ):
            tensor.copy_(baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark lookup and maintain directly with configurable request "
            "batch sizes. State restoration is excluded from NPU event timing."
        )
    )
    parser.add_argument(
        "--op",
        choices=("lookup", "maintain", "all"),
        default="all",
        help="operator to benchmark (default: all)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16),
        metavar="N",
        help="request batch sizes, separated by spaces (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--miss-count",
        type=int,
        default=QUERY_COUNT // 2,
        help=(
            "miss entries per request out of the fixed 2048 queries "
            "(default: 1024)"
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="untimed warmup iterations per case (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="timed iterations per case (default: 100)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="NPU device id (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260717,
        help="eviction seed passed to maintain (default: 20260717)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional path for configuration, summary, and raw timing samples",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[int]:
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than 0")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if not 0 <= args.miss_count <= QUERY_COUNT:
        raise ValueError(f"--miss-count must be in [0, {QUERY_COUNT}]")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("all --batch-sizes values must be greater than 0")
    return list(dict.fromkeys(args.batch_sizes))


def configure_custom_opp() -> tuple[Path, Path]:
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or spec.origin is None:
        raise RuntimeError("vllm_ascend is not installed in this environment")

    package_dir = Path(spec.origin).resolve().parent
    vendor_opp = (
        package_dir / "_cann_ops_custom" / "vendors" / "vllm-ascend"
    )
    aicpu_opp = vendor_opp / "op_impl" / "aicpu_transformer"
    if not vendor_opp.is_dir():
        raise RuntimeError(f"custom OPP directory does not exist: {vendor_opp}")
    if not aicpu_opp.is_dir():
        raise RuntimeError(f"AICPU OPP directory does not exist: {aicpu_opp}")

    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    custom_paths = f"{aicpu_opp}:{vendor_opp}"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{custom_paths}:{current}" if current else custom_paths
    )
    return package_dir, aicpu_opp


def load_custom_ops(torch) -> Path:
    import torch_npu  # noqa: F401
    import vllm_ascend.vllm_ascend_C as extension

    for op_name in (
        "asu_hbm_index_lookup",
        "asu_hbm_index_maintain_aicpu",
    ):
        if not hasattr(torch.ops._C_ascend, op_name):
            raise RuntimeError(
                f"PyTorch operator is not registered: _C_ascend::{op_name}"
            )
    return Path(extension.__file__).resolve()


def build_host_case(torch, batch_size: int, miss_count: int, op_name: str) -> HostCase:
    hit_count = QUERY_COUNT - miss_count
    resident_tokens = torch.arange(RESIDENT_COUNT, dtype=torch.int32)
    resident_slots = torch.arange(RESIDENT_COUNT, dtype=torch.int32)
    miss_tokens = torch.arange(
        MISS_TOKEN_BASE,
        MISS_TOKEN_BASE + miss_count,
        dtype=torch.int32,
    )
    allocated_slots = torch.arange(
        RESIDENT_COUNT,
        RESIDENT_COUNT + miss_count,
        dtype=torch.int32,
    )

    index_row = torch.full((INDEX_SIZE,), NOT_FOUND, dtype=torch.int32)
    index_row[resident_tokens.long()] = resident_slots
    slot_to_index_row = torch.full(
        (SLOT_COUNT,), NOT_FOUND, dtype=torch.int32
    )
    slot_to_index_row[:RESIDENT_COUNT] = resident_tokens
    free_slots_row = torch.arange(
        RESIDENT_COUNT, SLOT_COUNT, dtype=torch.int32
    )

    query_row = torch.cat((resident_tokens[:hit_count], miss_tokens))
    expected_slots_row = torch.cat(
        (resident_slots[:hit_count], allocated_slots)
    )
    expected_misses_row = torch.cat(
        (
            torch.zeros(hit_count, dtype=torch.int32),
            torch.ones(miss_count, dtype=torch.int32),
        )
    )

    if op_name == "maintain" and miss_count:
        index_row[miss_tokens.long()] = allocated_slots
        slot_to_index_row[allocated_slots.long()] = miss_tokens

    index = index_row.unsqueeze(0).repeat(batch_size, 1)
    slot_to_index = slot_to_index_row.unsqueeze(0).repeat(batch_size, 1)
    free_slots = free_slots_row.unsqueeze(0).repeat(batch_size, 1)
    initial_head = miss_count if op_name == "maintain" else 0
    free_head = torch.zeros(
        (batch_size, FREE_HEAD_STRIDE), dtype=torch.int32
    )
    free_head[:, 0].fill_(initial_head)
    req_pool_entries = torch.arange(batch_size, dtype=torch.int32)
    query_index = query_row.unsqueeze(0).repeat(batch_size, 1)
    expected_slots = expected_slots_row.unsqueeze(0).repeat(batch_size, 1)
    expected_misses = expected_misses_row.unsqueeze(0).repeat(batch_size, 1)

    return HostCase(
        mutable_state=(index, slot_to_index, free_slots, free_head),
        req_pool_entries=req_pool_entries,
        query_index=query_index,
        expected_slots=expected_slots,
        expected_misses=expected_misses,
    )


def move_case_to_device(host_case: HostCase, device: Any) -> DeviceCase:
    baseline_state = tuple(tensor.to(device) for tensor in host_case.mutable_state)
    return DeviceCase(
        baseline_state=baseline_state,
        mutable_state=tuple(tensor.clone() for tensor in baseline_state),
        req_pool_entries=host_case.req_pool_entries.to(device),
        query_index=host_case.query_index.to(device),
        expected_slots=host_case.expected_slots.to(device),
        expected_misses=host_case.expected_misses.to(device),
    )


def call_lookup(torch, case: DeviceCase, batch_size: int):
    index, slot_to_index, free_slots, free_head = case.mutable_state
    return torch.ops._C_ascend.asu_hbm_index_lookup(
        index,
        slot_to_index,
        free_slots,
        free_head,
        case.req_pool_entries,
        case.query_index,
        batch_size,
    )


def call_maintain(
    torch,
    case: DeviceCase,
    batch_size: int,
    seed: int,
) -> None:
    index, slot_to_index, free_slots, free_head = case.mutable_state
    torch.ops._C_ascend.asu_hbm_index_maintain_aicpu(
        index,
        slot_to_index,
        free_slots,
        free_head,
        case.req_pool_entries,
        case.expected_slots,
        batch_size,
        seed,
    )


def assert_equal(torch, name: str, actual: Any, expected: Any) -> None:
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    if torch.equal(actual_cpu, expected_cpu):
        return
    mismatch = (actual_cpu != expected_cpu).nonzero()
    first = tuple(int(value) for value in mismatch[0])
    raise AssertionError(
        f"{name} mismatch at {first}: actual={int(actual_cpu[first])}, "
        f"expected={int(expected_cpu[first])}, "
        f"total_mismatches={int(mismatch.shape[0])}"
    )


def check_case(
    torch,
    case: DeviceCase,
    op_name: str,
    batch_size: int,
    miss_count: int,
    seed: int,
) -> None:
    case.reset()
    if op_name == "lookup":
        slot_out, miss_out = call_lookup(torch, case, batch_size)
        torch.npu.synchronize()
        assert_equal(torch, "lookup slots", slot_out, case.expected_slots)
        assert_equal(torch, "lookup misses", miss_out, case.expected_misses)
        expected_head = torch.zeros(
            (batch_size, FREE_HEAD_STRIDE), dtype=torch.int32
        )
        expected_head[:, 0].fill_(miss_count)
        assert_equal(
            torch,
            "lookup free head",
            case.mutable_state[3],
            expected_head,
        )
        return

    call_maintain(torch, case, batch_size, seed)
    torch.npu.synchronize()
    expected_head = torch.zeros(
        (batch_size, FREE_HEAD_STRIDE), dtype=torch.int32
    )
    assert_equal(
        torch,
        "maintain free head",
        case.mutable_state[3],
        expected_head,
    )

    index, slot_to_index, _, _ = case.mutable_state
    row_ids = case.req_pool_entries.long().unsqueeze(1)
    query_slots = index[row_ids, case.query_index.long()]
    assert_equal(
        torch,
        "maintain protected token-to-slot state",
        query_slots,
        case.expected_slots,
    )
    slot_tokens = slot_to_index[row_ids, case.expected_slots.long()]
    assert_equal(
        torch,
        "maintain protected slot-to-token state",
        slot_tokens,
        case.query_index,
    )


def benchmark_case(
    torch,
    case: DeviceCase,
    invoke: Callable[[], Any],
    warmup_iterations: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup_iterations):
        case.reset()
        torch.npu.synchronize()
        output = invoke()
        torch.npu.synchronize()
        del output

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples_ms = []
    for _ in range(iterations):
        case.reset()
        torch.npu.synchronize()
        start.record()
        output = invoke()
        end.record()
        torch.npu.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))
        del output
    return samples_ms


def percentile(samples: list[float], ratio: float) -> float:
    ordered = sorted(samples)
    rank = max(0, math.ceil(len(ordered) * ratio) - 1)
    return ordered[rank]


def summarize(
    op_name: str,
    batch_size: int,
    miss_count: int,
    samples_ms: list[float],
) -> dict[str, Any]:
    mean_ms = statistics.fmean(samples_ms)
    return {
        "operator": op_name,
        "batch_size": batch_size,
        "query_count_per_request": QUERY_COUNT,
        "hit_count_per_request": QUERY_COUNT - miss_count,
        "miss_count_per_request": miss_count,
        "min_ms": min(samples_ms),
        "mean_ms": mean_ms,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "max_ms": max(samples_ms),
        "requests_per_second": batch_size * 1000.0 / mean_ms,
        "query_items_per_second": batch_size * QUERY_COUNT * 1000.0 / mean_ms,
        "samples_ms": samples_ms,
    }


def print_result(result: dict[str, Any]) -> None:
    print(
        f"{result['operator']:<8} "
        f"batch={result['batch_size']:<4d} "
        f"mean={result['mean_ms']:>9.3f} ms "
        f"median={result['median_ms']:>9.3f} ms "
        f"p95={result['p95_ms']:>9.3f} ms "
        f"min={result['min_ms']:>9.3f} ms "
        f"req/s={result['requests_per_second']:>11.2f}"
    )


def main() -> None:
    args = parse_args()
    batch_sizes = validate_args(args)
    package_dir, aicpu_opp = configure_custom_opp()

    import torch

    extension_path = load_custom_ops(torch)
    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)
    torch.set_grad_enabled(False)
    operators = ("lookup", "maintain") if args.op == "all" else (args.op,)

    print(f"[INFO] vllm_ascend package={package_dir}")
    print(f"[INFO] extension={extension_path}")
    print(f"[INFO] AICPU OPP={aicpu_opp}")
    print(f"[INFO] device={device}")
    print(
        f"[INFO] operators={operators}, batch_sizes={batch_sizes}, "
        f"misses/request={args.miss_count}, warmup={args.warmup_iterations}, "
        f"iterations={args.iterations}"
    )

    results = []
    for op_name in operators:
        for batch_size in batch_sizes:
            host_case = build_host_case(
                torch, batch_size, args.miss_count, op_name
            )
            case = move_case_to_device(host_case, device)
            check_case(
                torch,
                case,
                op_name,
                batch_size,
                args.miss_count,
                args.seed,
            )

            if op_name == "lookup":
                invoke = partial(call_lookup, torch, case, batch_size)
            else:
                invoke = partial(
                    call_maintain, torch, case, batch_size, args.seed
                )

            samples_ms = benchmark_case(
                torch,
                case,
                invoke,
                args.warmup_iterations,
                args.iterations,
            )
            result = summarize(
                op_name, batch_size, args.miss_count, samples_ms
            )
            results.append(result)
            print_result(result)

            del case, host_case, invoke
            gc.collect()
            torch.npu.empty_cache()

    if args.output_json:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "vllm_ascend_package": str(package_dir),
            "extension": str(extension_path),
            "aicpu_opp": str(aicpu_opp),
            "configuration": {
                "operators": operators,
                "batch_sizes": batch_sizes,
                "miss_count_per_request": args.miss_count,
                "query_count_per_request": QUERY_COUNT,
                "warmup_iterations": args.warmup_iterations,
                "iterations": args.iterations,
                "seed": args.seed,
            },
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[PASS] wrote benchmark report to {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
