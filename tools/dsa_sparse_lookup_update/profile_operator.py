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
    validate_dimensions,
)

CUSTOM_OP_NAMES = (
    "dsasparselookupupdate",
    "dsa_sparse_lookup_update",
    "aclnndsasparselookupupdate",
)
PROFILE_FILENAMES = {
    "kernel_details.csv",
    "operator_details.csv",
    "op_statistic.csv",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Event-time and trace the standalone Ascend 950 "
            "dsa_sparse_lookup_update operator."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--scenario", choices=("steady", "fresh"), default="steady")
    parser.add_argument("--seats", type=int, default=8)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--slots", type=int, default=4096)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-iters", type=int, default=20)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _make_epoch_tensors(
    runtime: Runtime,
    *,
    rows: int,
    count: int,
    first_epoch: int,
) -> list[Any]:
    torch = runtime.torch
    return [
        torch.full(
            (rows,),
            first_epoch + index,
            dtype=torch.int32,
            device=runtime.device,
        )
        for index in range(count)
    ]


def _invoke_scenario(
    runtime: Runtime,
    inputs: OperatorInputs,
    *,
    epoch_tensor: Any | None,
) -> None:
    if epoch_tensor is not None:
        inputs.row_seat_epoch = epoch_tensor
    invoke(runtime, inputs)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[index]


def _summarize_us(samples_us: list[float]) -> dict[str, float | int]:
    sorted_samples = sorted(samples_us)
    return {
        "count": len(sorted_samples),
        "min_us": sorted_samples[0],
        "median_us": statistics.median(sorted_samples),
        "p95_us": _percentile(sorted_samples, 0.95),
        "max_us": sorted_samples[-1],
        "mean_us": statistics.fmean(sorted_samples),
    }


def _event_benchmark(
    runtime: Runtime,
    inputs: OperatorInputs,
    *,
    scenario: str,
    warmup: int,
    iterations: int,
    epoch_tensors: list[Any],
) -> dict[str, float | int]:
    torch = runtime.torch
    epoch_index = 0

    for _ in range(warmup):
        epoch_tensor = epoch_tensors[epoch_index] if scenario == "fresh" else None
        _invoke_scenario(runtime, inputs, epoch_tensor=epoch_tensor)
        epoch_index += scenario == "fresh"
    torch.npu.synchronize()

    start_events = [
        torch.npu.Event(enable_timing=True)
        for _ in range(iterations)
    ]
    end_events = [
        torch.npu.Event(enable_timing=True)
        for _ in range(iterations)
    ]
    for iteration in range(iterations):
        epoch_tensor = epoch_tensors[epoch_index] if scenario == "fresh" else None
        start_events[iteration].record()
        _invoke_scenario(runtime, inputs, epoch_tensor=epoch_tensor)
        end_events[iteration].record()
        epoch_index += scenario == "fresh"
    torch.npu.synchronize()

    samples_us = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(start_events, end_events)
    ]
    return _summarize_us(samples_us)


def _create_profiler(runtime: Runtime, trace_dir: Path) -> Any:
    profiler_api = runtime.torch_npu.profiler
    experimental_config = profiler_api._ExperimentalConfig(
        export_type=profiler_api.ExportType.Text,
        profiler_level=profiler_api.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=profiler_api.AiCMetrics.PipeUtilization,
        l2_cache=False,
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
            worker_name="dsa_sparse_lookup_update",
            analyse_flag=True,
            async_mode=False,
        ),
    )


def _trace(
    runtime: Runtime,
    inputs: OperatorInputs,
    *,
    scenario: str,
    iterations: int,
    epoch_tensors: list[Any],
    trace_dir: Path,
) -> None:
    profiler = _create_profiler(runtime, trace_dir)
    profiler.start()
    for iteration in range(iterations):
        epoch_tensor = epoch_tensors[iteration] if scenario == "fresh" else None
        _invoke_scenario(runtime, inputs, epoch_tensor=epoch_tensor)
        profiler.step()
    runtime.torch.npu.synchronize()
    profiler.stop()


def _profile_contains_custom_op(trace_dir: Path) -> bool:
    profile_files = [
        path
        for path in trace_dir.rglob("*")
        if path.is_file() and path.name in PROFILE_FILENAMES
    ]
    if not profile_files:
        raise RuntimeError(
            "Profiler parsing completed without kernel_details.csv, "
            f"operator_details.csv, or op_statistic.csv under {trace_dir}."
        )
    for profile_file in profile_files:
        with profile_file.open(encoding="utf-8", errors="replace") as csv_file:
            for line in csv_file:
                lowered = line.lower()
                if any(operator_name in lowered for operator_name in CUSTOM_OP_NAMES):
                    return True
    return False


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


def main() -> int:
    args = _parse_args()
    validate_dimensions(
        seats=args.seats,
        rows=args.rows,
        max_model_len=args.max_model_len,
        slots=args.slots,
        lanes=args.lanes,
        topk=args.topk,
    )
    for name in ("warmup", "iterations", "profile_iters"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive, got {value}.")

    runtime = load_runtime(
        device=args.device,
        install_root=args.install_root,
    )
    inputs = make_profile_inputs(
        runtime,
        seats=args.seats,
        rows=args.rows,
        max_model_len=args.max_model_len,
        slots=args.slots,
        lanes=args.lanes,
        topk=args.topk,
    )

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else TOOL_DIR / "profiles" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    event_epoch_count = args.warmup + args.iterations
    event_epochs = (
        _make_epoch_tensors(
            runtime,
            rows=args.rows,
            count=event_epoch_count,
            first_epoch=1,
        )
        if args.scenario == "fresh"
        else []
    )
    runtime.torch.npu.synchronize()

    with runtime.torch.inference_mode():
        timing = _event_benchmark(
            runtime,
            inputs,
            scenario=args.scenario,
            warmup=args.warmup,
            iterations=args.iterations,
            epoch_tensors=event_epochs,
        )

        trace_dir: Path | None = None
        custom_op_found: bool | None = None
        if not args.no_trace:
            trace_dir = output_dir / "trace"
            trace_dir.mkdir()
            trace_epochs = (
                _make_epoch_tensors(
                    runtime,
                    rows=args.rows,
                    count=args.profile_iters,
                    first_epoch=event_epoch_count + 1,
                )
                if args.scenario == "fresh"
                else []
            )
            runtime.torch.npu.synchronize()
            _trace(
                runtime,
                inputs,
                scenario=args.scenario,
                iterations=args.profile_iters,
                epoch_tensors=trace_epochs,
                trace_dir=trace_dir,
            )
            custom_op_found = _profile_contains_custom_op(trace_dir)

    manifest = {
        "operator": "dsa_sparse_lookup_update",
        "scenario": args.scenario,
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
        "shape": {
            "seats": args.seats,
            "rows": args.rows,
            "max_model_len": args.max_model_len,
            "slots": args.slots,
            "lanes": args.lanes,
            "topk": args.topk,
        },
        "measurement": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "profile_iterations": 0 if args.no_trace else args.profile_iters,
            **timing,
        },
        "trace_dir": str(trace_dir) if trace_dir is not None else None,
        "trace_contains_custom_op": custom_op_found,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Artifacts: {output_dir}")
    if custom_op_found is False:
        raise RuntimeError(
            "The parsed profile does not contain DsaSparseLookupUpdate, "
            "dsa_sparse_lookup_update, or aclnnDsaSparseLookupUpdate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
