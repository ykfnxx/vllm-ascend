#!/usr/bin/env python3
"""Compare eager and ACL NPU Graph execution of one Maintain operator.

Both modes use the same operator binary, tensor addresses, stream, reset
operation, seed, NPU Event timing, and warmup policy. The only intended
difference is whether the operator is launched directly or through
``torch.npu.NPUGraph.replay()``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import benchmark_asu_hbm_index_ops as benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare isolated eager and ACL NPU Graph execution of "
            "asu_hbm_index_maintain_aicpu."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="request count passed as req_num (default: 32)",
    )
    parser.add_argument(
        "--miss-count",
        type=int,
        default=300,
        help=(
            "dynamic Maintain eviction count used to initialize free_head "
            "(default: 300; ignored by fixed-workload kernels)"
        ),
    )
    parser.add_argument(
        "--capture-warmup-iterations",
        type=int,
        default=3,
        help="eager calls issued on the graph stream before capture (default: 3)",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="unmeasured calls or replays per execution mode (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="NPU Event timing samples per execution mode (default: 100)",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=Path,
        help=(
            "optional empty directory for separate eager and graph Level2 "
            "torch-npu profiles"
        ),
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=20,
        help="calls or replays captured per profile mode (default: 20)",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help=(
            "skip dynamic Maintain state checks; required for the synthetic "
            "fixed-workload kernel"
        ),
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
        help="seed passed to Maintain in both modes (default: 20260717)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional path for configuration, summaries, and raw samples",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if not 0 <= args.miss_count <= benchmark.QUERY_COUNT:
        raise ValueError(
            f"--miss-count must be in [0, {benchmark.QUERY_COUNT}]"
        )
    if args.capture_warmup_iterations < 0:
        raise ValueError("--capture-warmup-iterations cannot be negative")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than 0")
    if args.profile_iterations <= 0:
        raise ValueError("--profile-iterations must be greater than 0")


def reset_case(torch: Any, case: Any, stream: Any) -> None:
    with torch.npu.stream(stream):
        case.reset()
    stream.synchronize()


def launch_maintain(
    torch: Any,
    case: Any,
    stream: Any,
    batch_size: int,
    seed: int,
) -> None:
    with torch.npu.stream(stream):
        benchmark.call_maintain(torch, case, batch_size, seed)


def capture_maintain_graph(
    torch: Any,
    case: Any,
    stream: Any,
    batch_size: int,
    seed: int,
    capture_warmup_iterations: int,
) -> Any:
    for _ in range(capture_warmup_iterations):
        reset_case(torch, case, stream)
        launch_maintain(torch, case, stream, batch_size, seed)
        stream.synchronize()

    reset_case(torch, case, stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        with torch.npu.graph(graph):
            benchmark.call_maintain(torch, case, batch_size, seed)
    stream.synchronize()
    return graph


def measure_mode(
    torch: Any,
    case: Any,
    stream: Any,
    launch: Callable[[], None],
    warmup_iterations: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup_iterations):
        reset_case(torch, case, stream)
        with torch.npu.stream(stream):
            launch()
        stream.synchronize()

    samples_ms: list[float] = []
    for _ in range(iterations):
        reset_case(torch, case, stream)
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        with torch.npu.stream(stream):
            start.record()
            launch()
            end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))
    return samples_ms


def validate_dynamic_maintain_state(
    torch: Any,
    case: Any,
    batch_size: int,
) -> None:
    expected_head = torch.zeros(
        (batch_size, benchmark.FREE_HEAD_STRIDE), dtype=torch.int32
    )
    benchmark.assert_equal(
        torch,
        "graph maintain free head",
        case.mutable_state[3],
        expected_head,
    )

    index, slot_to_index, _, _ = case.mutable_state
    row_ids = case.req_pool_entries.long().unsqueeze(1)
    query_slots = index[row_ids, case.query_index.long()]
    benchmark.assert_equal(
        torch,
        "graph maintain protected token-to-slot state",
        query_slots,
        case.expected_slots,
    )
    slot_tokens = slot_to_index[row_ids, case.expected_slots.long()]
    benchmark.assert_equal(
        torch,
        "graph maintain protected slot-to-token state",
        slot_tokens,
        case.query_index,
    )


def prepare_profile_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"--profile-output-dir is not a directory: {root}")
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            "--profile-output-dir must be empty to avoid mixing traces: "
            f"{root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def profile_mode(
    torch: Any,
    torch_npu: Any,
    case: Any,
    stream: Any,
    launch: Callable[[], None],
    iterations: int,
    output_dir: Path,
    mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(output_dir),
        worker_name=f"asu_maintain_{mode}",
        analyse_flag=True,
        async_mode=False,
    )
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        experimental_config=experimental_config,
        on_trace_ready=handler,
    ):
        for _ in range(iterations):
            reset_case(torch, case, stream)
            with torch.npu.stream(stream):
                launch()
            stream.synchronize()


def summarize_mode(
    mode: str,
    batch_size: int,
    miss_count: int,
    samples_ms: list[float],
) -> dict[str, Any]:
    result = benchmark.summarize(
        f"maintain_{mode}", batch_size, miss_count, samples_ms
    )
    result["mode"] = mode
    return result


def print_summary(result: dict[str, Any]) -> None:
    print(
        f"[RESULT] mode={result['mode']} "
        f"mean={result['mean_ms'] * 1000.0:.3f}us "
        f"median={result['median_ms'] * 1000.0:.3f}us "
        f"p95={result['p95_ms'] * 1000.0:.3f}us "
        f"min={result['min_ms'] * 1000.0:.3f}us "
        f"max={result['max_ms'] * 1000.0:.3f}us"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    package_dir, aicpu_opp = benchmark.configure_custom_opp()

    import torch
    import torch_npu

    extension_path = benchmark.load_custom_ops(torch)
    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)
    torch.set_grad_enabled(False)

    host_case = benchmark.build_host_case(
        torch, args.batch_size, args.miss_count, "maintain"
    )
    case = benchmark.move_case_to_device(host_case, device)
    stream = torch.npu.Stream(device=device)

    if not args.skip_check:
        benchmark.check_case(
            torch,
            case,
            "maintain",
            args.batch_size,
            args.miss_count,
            args.seed,
        )

    graph = capture_maintain_graph(
        torch,
        case,
        stream,
        args.batch_size,
        args.seed,
        args.capture_warmup_iterations,
    )
    graph_launch = graph.replay
    if not args.skip_check:
        reset_case(torch, case, stream)
        with torch.npu.stream(stream):
            graph_launch()
        stream.synchronize()
        validate_dynamic_maintain_state(torch, case, args.batch_size)

    print(f"[INFO] vllm_ascend package={package_dir}")
    print(f"[INFO] extension={extension_path}")
    print(f"[INFO] AICPU OPP={aicpu_opp}")
    print(f"[INFO] device={device}")
    print(
        f"[INFO] batch_size={args.batch_size}, "
        f"miss_count={args.miss_count}, seed={args.seed}"
    )
    print(
        f"[INFO] capture_warmup={args.capture_warmup_iterations}, "
        f"warmup={args.warmup_iterations}, iterations={args.iterations}"
    )

    eager_launch = partial(
        benchmark.call_maintain,
        torch,
        case,
        args.batch_size,
        args.seed,
    )
    eager_samples = measure_mode(
        torch,
        case,
        stream,
        eager_launch,
        args.warmup_iterations,
        args.iterations,
    )
    graph_samples = measure_mode(
        torch,
        case,
        stream,
        graph_launch,
        args.warmup_iterations,
        args.iterations,
    )

    eager_result = summarize_mode(
        "eager", args.batch_size, args.miss_count, eager_samples
    )
    graph_result = summarize_mode(
        "graph", args.batch_size, args.miss_count, graph_samples
    )
    print_summary(eager_result)
    print_summary(graph_result)
    graph_to_eager = (
        statistics.fmean(graph_samples) / statistics.fmean(eager_samples)
    )
    print(f"[RESULT] graph/eager mean ratio={graph_to_eager:.3f}x")

    profile_root = None
    if args.profile_output_dir is not None:
        profile_root = prepare_profile_root(args.profile_output_dir)
        profile_mode(
            torch,
            torch_npu,
            case,
            stream,
            eager_launch,
            args.profile_iterations,
            profile_root / "eager",
            "eager",
        )
        profile_mode(
            torch,
            torch_npu,
            case,
            stream,
            graph_launch,
            args.profile_iterations,
            profile_root / "graph",
            "graph",
        )
        print(f"[PASS] wrote separate eager/graph profiles to {profile_root}")

    if args.output_json is not None:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "vllm_ascend_package": str(package_dir),
            "extension": str(extension_path),
            "aicpu_opp": str(aicpu_opp),
            "configuration": {
                "batch_size": args.batch_size,
                "miss_count": args.miss_count,
                "seed": args.seed,
                "capture_warmup_iterations": args.capture_warmup_iterations,
                "warmup_iterations": args.warmup_iterations,
                "iterations": args.iterations,
                "profile_iterations": args.profile_iterations,
                "profile_output_dir": (
                    str(profile_root) if profile_root is not None else None
                ),
                "skip_check": args.skip_check,
            },
            "eager": eager_result,
            "graph": graph_result,
            "graph_to_eager_mean_ratio": graph_to_eager,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[PASS] wrote comparison report to {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
