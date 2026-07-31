#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from common import (
    QUERY_COUNT,
    RESIDENT_SLOT_COUNT,
    TOOL_DIR,
    OperatorInputs,
    Runtime,
    clone_operator_inputs,
    invoke,
    load_runtime,
    make_profile_inputs,
    restore_operator_inputs,
    validate_requests,
)
from profile_operator import (
    _device_name,
    _event_benchmark,
    _git_head,
    _profile_contains_custom_op,
)

DEFAULT_REQUEST_COUNTS = (32,)
DEFAULT_MISS_COUNTS = (0, 1, 205, QUERY_COUNT)


@dataclass(frozen=True)
class MetricProfile:
    name: str
    enum_attribute: str
    collect_l2_cache: bool = False


@dataclass(frozen=True)
class Workload:
    requests: int
    miss_count: int

    @property
    def name(self) -> str:
        return f"req-{self.requests:04d}_miss-{self.miss_count:04d}"


METRIC_PROFILES = (
    MetricProfile("pipe-utilization", "PipeUtilization"),
    MetricProfile("memory", "Memory"),
    MetricProfile("resource-conflict", "ResourceConflictRatio"),
    MetricProfile("l2-cache", "L2Cache", collect_l2_cache=True),
)
METRIC_NAMES = tuple(profile.name for profile in METRIC_PROFILES)


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one DSA sparse profiling command that emits independent "
            "Event measurements and multiple torch_npu.profiler traces."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument(
        "--requests",
        type=int,
        nargs="+",
        default=list(DEFAULT_REQUEST_COUNTS),
        metavar="N",
        help="One or more request counts. Default: 32.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-iters", type=int, default=20)
    miss_group = parser.add_mutually_exclusive_group()
    miss_group.add_argument(
        "--miss-counts",
        type=int,
        nargs="+",
        metavar="N",
        help=(
            "Exact misses per 2K query. "
            "Default: 0 1 205 2048."
        ),
    )
    miss_group.add_argument(
        "--miss-rates",
        type=float,
        nargs="+",
        metavar="PERCENT",
        help=(
            "Miss percentages converted to the nearest query count. "
            "Mutually exclusive with --miss-counts."
        ),
    )
    parser.add_argument(
        "--metrics",
        choices=METRIC_NAMES,
        nargs="+",
        default=list(METRIC_NAMES),
        help="Independent profiler metric groups to collect.",
    )
    parser.add_argument(
        "--skip-event",
        action="store_true",
        help="Skip the NPU Event benchmark and emit traces only.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _deduplicate(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(values))


def _miss_counts(args: argparse.Namespace) -> tuple[int, ...]:
    if args.miss_rates is not None:
        counts: list[int] = []
        for miss_rate in args.miss_rates:
            if not 0.0 <= miss_rate <= 100.0:
                raise ValueError(
                    "Every miss rate must be in [0, 100], "
                    f"got {miss_rate}."
                )
            counts.append(
                math.floor(
                    QUERY_COUNT * miss_rate / 100.0 + 0.5
                )
            )
    elif args.miss_counts is not None:
        counts = list(args.miss_counts)
    else:
        counts = list(DEFAULT_MISS_COUNTS)

    for miss_count in counts:
        if not 0 <= miss_count <= QUERY_COUNT:
            raise ValueError(
                "Every miss count must be in "
                f"[0, {QUERY_COUNT}], got {miss_count}."
            )
    return _deduplicate(counts)


def _workloads(args: argparse.Namespace) -> tuple[Workload, ...]:
    request_counts = _deduplicate(args.requests)
    for requests in request_counts:
        validate_requests(requests)
    return tuple(
        Workload(requests=requests, miss_count=miss_count)
        for requests in request_counts
        for miss_count in _miss_counts(args)
    )


def _metric_profiles(
    names: Sequence[str],
) -> tuple[MetricProfile, ...]:
    profiles_by_name = {
        profile.name: profile
        for profile in METRIC_PROFILES
    }
    return tuple(
        profiles_by_name[name]
        for name in dict.fromkeys(names)
    )


def _validate_iterations(args: argparse.Namespace) -> None:
    for name in ("warmup", "iterations", "profile_iters"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(
                f"{name.replace('_', '-')} must be positive, "
                f"got {value}."
            )


def _create_profiler(
    runtime: Runtime,
    trace_dir: Path,
    metric: MetricProfile,
) -> Any:
    profiler_api = runtime.torch_npu.profiler
    try:
        aic_metric = getattr(
            profiler_api.AiCMetrics,
            metric.enum_attribute,
        )
    except AttributeError as error:
        raise RuntimeError(
            "The installed torch_npu does not provide "
            f"AiCMetrics.{metric.enum_attribute}."
        ) from error

    experimental_config = profiler_api._ExperimentalConfig(
        export_type=profiler_api.ExportType.Text,
        profiler_level=profiler_api.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=aic_metric,
        l2_cache=metric.collect_l2_cache,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    return profiler_api.profile(
        activities=[
            profiler_api.ProfilerActivity.CPU,
            profiler_api.ProfilerActivity.NPU,
        ],
        with_stack=False,
        profile_memory=False,
        with_modules=False,
        experimental_config=experimental_config,
        on_trace_ready=profiler_api.tensorboard_trace_handler(
            str(trace_dir),
            worker_name=(
                "dsa_sparse_lookup_update_"
                f"{metric.name.replace('-', '_')}"
            ),
            analyse_flag=True,
            async_mode=False,
        ),
    )


def _trace(
    runtime: Runtime,
    inputs: OperatorInputs,
    reference: OperatorInputs,
    *,
    reset_state: bool,
    iterations: int,
    trace_dir: Path,
    metric: MetricProfile,
) -> None:
    profiler = _create_profiler(runtime, trace_dir, metric)
    profiler.start()
    for _ in range(iterations):
        if reset_state:
            restore_operator_inputs(inputs, reference)
        invoke(runtime, inputs)
        profiler.step()
    runtime.torch.npu.synchronize()
    profiler.stop()


def _profile_files(trace_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(trace_dir))
        for path in trace_dir.rglob("*")
        if path.is_file()
        and path.name
        in {
            "kernel_details.csv",
            "operator_details.csv",
            "op_statistic.csv",
            "l2_cache.csv",
        }
    )


def _git_branch() -> str | None:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=TOOL_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip()
    return branch if result.returncode == 0 and branch else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_workload(
    runtime: Runtime,
    workload: Workload,
    metrics: tuple[MetricProfile, ...],
    *,
    output_dir: Path,
    warmup: int,
    iterations: int,
    profile_iterations: int,
    skip_event: bool,
) -> dict[str, Any]:
    reference = make_profile_inputs(
        runtime,
        requests=workload.requests,
        miss_count=workload.miss_count,
    )
    inputs = clone_operator_inputs(reference)
    reset_state = workload.miss_count > 0
    workload_dir = output_dir / workload.name
    workload_dir.mkdir()

    event_timing: dict[str, float | int] | None = None
    if not skip_event:
        event_timing = _event_benchmark(
            runtime,
            inputs,
            reference,
            reset_state=reset_state,
            warmup=warmup,
            iterations=iterations,
        )
        print(
            f"[{workload.name}] Event median: "
            f"{event_timing['median_us']:.3f} us"
        )

    profiles: dict[str, Any] = {}
    for metric in metrics:
        trace_dir = workload_dir / metric.name
        trace_dir.mkdir()
        print(
            f"[{workload.name}] Profiling {metric.name} "
            f"for {profile_iterations} iterations..."
        )
        _trace(
            runtime,
            inputs,
            reference,
            reset_state=reset_state,
            iterations=profile_iterations,
            trace_dir=trace_dir,
            metric=metric,
        )
        custom_op_found = _profile_contains_custom_op(trace_dir)
        if not custom_op_found:
            raise RuntimeError(
                "The parsed profile does not contain "
                "DsaSparseLookupUpdate, dsa_sparse_lookup_update, "
                "or aclnnDsaSparseLookupUpdate."
            )
        profiles[metric.name] = {
            "aic_metric": metric.enum_attribute,
            "collect_l2_cache": metric.collect_l2_cache,
            "trace_dir": str(trace_dir),
            "parsed_files": _profile_files(trace_dir),
            "trace_contains_custom_op": True,
        }

    workload_manifest: dict[str, Any] = {
        "name": workload.name,
        "requests": workload.requests,
        "resident_entries_per_request": RESIDENT_SLOT_COUNT,
        "query_width": QUERY_COUNT,
        "miss_count": workload.miss_count,
        "effective_miss_rate_percent": (
            100.0 * workload.miss_count / QUERY_COUNT
        ),
        "state_reset_excluded_from_event_timing": reset_state,
        "trace_contains_state_reset": reset_state,
        "event_timing": event_timing,
        "profiles": profiles,
    }
    _write_json(
        workload_dir / "manifest.json",
        workload_manifest,
    )
    return workload_manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_iterations(args)
    workloads = _workloads(args)
    metrics = _metric_profiles(args.metrics)

    runtime = load_runtime(
        device=args.device,
        install_root=args.install_root,
    )
    timestamp = (
        datetime.now()
        .astimezone()
        .strftime("%Y%m%d-%H%M%S")
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else TOOL_DIR / "profiles" / f"matrix-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    runtime.torch.npu.synchronize()
    results: list[dict[str, Any]] = []
    with runtime.torch.inference_mode():
        for workload in workloads:
            results.append(
                _run_workload(
                    runtime,
                    workload,
                    metrics,
                    output_dir=output_dir,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    profile_iterations=args.profile_iters,
                    skip_event=args.skip_event,
                )
            )

    manifest = {
        "operator": "dsa_sparse_lookup_update",
        "scenario": "profile-matrix",
        "device": args.device,
        "device_name": _device_name(runtime),
        "git_branch": _git_branch(),
        "git_head": _git_head(),
        "torch_version": runtime.torch.__version__,
        "torch_npu_version": runtime.torch_npu.__version__,
        "install_root": (
            str(runtime.install_root)
            if runtime.install_root is not None
            else None
        ),
        "measurement": {
            "warmup": args.warmup,
            "event_iterations": (
                0 if args.skip_event else args.iterations
            ),
            "profile_iterations": args.profile_iters,
            "metric_profiles": [
                metric.name for metric in metrics
            ],
        },
        "workloads": results,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Profile matrix manifest: {manifest_path}")
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
