#!/usr/bin/env python3
"""Profile one ASU HBM index custom operator and parse the trace.

Each invocation profiles exactly one of ``asu_hbm_index_lookup`` and
``asu_hbm_index_maintain_aicpu``. Tensor construction and warmup happen before
profiling starts. Once profiling stops, torch-npu synchronously parses the raw
trace and the script copies the parsed artifacts into a separate directory.

Examples:

    python3 examples/dsa_sparse/profile_asu_hbm_index_ops.py \
        --op lookup --batch-size 8 \
        --output-dir /data/asu-profiles/lookup-bs8

    python3 examples/dsa_sparse/profile_asu_hbm_index_ops.py \
        --op maintain --batch-size 8 \
        --output-dir /data/asu-profiles/maintain-bs8
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INDEX_SIZE = 128 * 1024
SLOT_COUNT = 10 * 1024
FREE_SLOT_COUNT = 2 * 1024
QUERY_COUNT = 2 * 1024
RESIDENT_COUNT = SLOT_COUNT - FREE_SLOT_COUNT
FIXED_UPDATE_COUNT = 300
FIXED_HIT_COUNT = QUERY_COUNT - FIXED_UPDATE_COUNT
FREE_HEAD_STRIDE = 16
MISS_TOKEN_BASE = INDEX_SIZE // 2
NOT_FOUND = -1


@dataclass(frozen=True)
class OperatorCase:
    baseline_state: tuple[Any, Any, Any, Any] | None
    index: Any
    slot_to_index: Any
    free_slots: Any
    free_head: Any
    req_pool_entries: Any
    query_index: Any
    last_query_slots: Any

    def reset(self) -> None:
        if self.baseline_state is None:
            raise RuntimeError("reset requested without an NPU baseline")
        for tensor, baseline in zip(
            (
                self.index,
                self.slot_to_index,
                self.free_slots,
                self.free_head,
            ),
            self.baseline_state,
            strict=True,
        ):
            tensor.copy_(baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one ASU HBM index custom operator, synchronously parse "
            "the trace, and copy parsed artifacts into a separate directory."
        )
    )
    parser.add_argument(
        "--op",
        choices=("lookup", "maintain"),
        required=True,
        help="single operator to profile",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="request count passed as req_num (default: 1)",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="unprofiled warmup calls (default: 10)",
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=20,
        help="calls captured by the profiler (default: 20)",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help=(
            "restore the four mutable state tensors and synchronize before "
            "every warmup/profile call, matching the benchmark methodology"
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
        help="seed passed to the maintain operator (default: 20260717)",
    )
    parser.add_argument(
        "--export-type",
        choices=("db", "text"),
        default="db",
        help="parsed torch-npu export format (default: db)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="empty directory in which raw and parsed results are written",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if args.profile_iterations <= 0:
        raise ValueError("--profile-iterations must be greater than 0")
    if args.device_id < 0:
        raise ValueError("--device-id cannot be negative")


def prepare_output_dir(output_dir: Path) -> tuple[Path, Path, Path]:
    run_root = output_dir.expanduser().resolve()
    if run_root.exists() and not run_root.is_dir():
        raise RuntimeError(f"--output-dir is not a directory: {run_root}")
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(
            f"--output-dir must be empty to avoid mixing profiles: {run_root}"
        )
    raw_root = run_root / "raw"
    parsed_root = run_root / "parsed"
    raw_root.mkdir(parents=True, exist_ok=True)
    parsed_root.mkdir(parents=True, exist_ok=True)
    return run_root, raw_root, parsed_root


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


def load_custom_ops(torch: Any) -> Path:
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


def build_case(
    torch: Any,
    device: Any,
    batch_size: int,
    reset_state: bool,
) -> OperatorCase:
    resident_tokens = torch.arange(RESIDENT_COUNT, dtype=torch.int32)
    index = torch.full(
        (batch_size, INDEX_SIZE), NOT_FOUND, dtype=torch.int32
    )
    index[:, :RESIDENT_COUNT] = resident_tokens

    slot_to_index = torch.full(
        (batch_size, SLOT_COUNT), NOT_FOUND, dtype=torch.int32
    )
    slot_to_index[:, :RESIDENT_COUNT] = resident_tokens

    free_slots_row = torch.arange(
        RESIDENT_COUNT, SLOT_COUNT, dtype=torch.int32
    )
    free_slots = free_slots_row.unsqueeze(0).repeat(batch_size, 1)
    free_head = torch.zeros(
        (batch_size, FREE_HEAD_STRIDE), dtype=torch.int32
    )
    req_pool_entries = torch.arange(batch_size, dtype=torch.int32)

    hit_queries = torch.arange(FIXED_HIT_COUNT, dtype=torch.int32)
    update_queries = torch.arange(
        MISS_TOKEN_BASE,
        MISS_TOKEN_BASE + FIXED_UPDATE_COUNT,
        dtype=torch.int32,
    )
    query_row = torch.cat((hit_queries, update_queries))
    query_index = query_row.unsqueeze(0).repeat(batch_size, 1)

    hit_slots = torch.arange(FIXED_HIT_COUNT, dtype=torch.int32)
    update_slots = torch.arange(
        RESIDENT_COUNT,
        RESIDENT_COUNT + FIXED_UPDATE_COUNT,
        dtype=torch.int32,
    )
    last_query_slots_row = torch.cat((hit_slots, update_slots))
    last_query_slots = last_query_slots_row.unsqueeze(0).repeat(
        batch_size, 1
    )

    device_state = tuple(
        tensor.to(device)
        for tensor in (index, slot_to_index, free_slots, free_head)
    )
    baseline_state = device_state if reset_state else None
    mutable_state = (
        tuple(tensor.clone() for tensor in device_state)
        if reset_state
        else device_state
    )
    return OperatorCase(
        baseline_state=baseline_state,
        index=mutable_state[0],
        slot_to_index=mutable_state[1],
        free_slots=mutable_state[2],
        free_head=mutable_state[3],
        req_pool_entries=req_pool_entries.to(device),
        query_index=query_index.to(device),
        last_query_slots=last_query_slots.to(device),
    )


def invoke_lookup(torch: Any, case: OperatorCase, batch_size: int) -> Any:
    return torch.ops._C_ascend.asu_hbm_index_lookup(
        case.index,
        case.slot_to_index,
        case.free_slots,
        case.free_head,
        case.req_pool_entries,
        case.query_index,
        batch_size,
    )


def invoke_maintain(
    torch: Any,
    case: OperatorCase,
    batch_size: int,
    seed: int,
) -> None:
    torch.ops._C_ascend.asu_hbm_index_maintain_aicpu(
        case.index,
        case.slot_to_index,
        case.free_slots,
        case.free_head,
        case.req_pool_entries,
        case.last_query_slots,
        batch_size,
        seed,
    )


def validate_lookup_output(output: Any, batch_size: int) -> None:
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError("lookup did not return the expected two tensors")
    expected_shape = (batch_size, QUERY_COUNT)
    for name, tensor in zip(("slot_out", "miss_out"), output, strict=True):
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"lookup {name} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}"
            )


def warmup(
    torch: Any,
    case: OperatorCase,
    op_name: str,
    batch_size: int,
    seed: int,
    iterations: int,
    reset_state: bool,
) -> None:
    last_output = None
    for _ in range(iterations):
        if reset_state:
            case.reset()
            torch.npu.synchronize()
        if op_name == "lookup":
            last_output = invoke_lookup(torch, case, batch_size)
        else:
            invoke_maintain(torch, case, batch_size, seed)
        if reset_state:
            torch.npu.synchronize()
    torch.npu.synchronize()
    if op_name == "lookup" and last_output is not None:
        validate_lookup_output(last_output, batch_size)
    del last_output
    gc.collect()


def create_experimental_config(torch_npu: Any, op_name: str, export_type: str) -> Any:
    profiler_level = (
        torch_npu.profiler.ProfilerLevel.Level1
        if op_name == "lookup"
        else torch_npu.profiler.ProfilerLevel.Level2
    )
    profiler_export_type = (
        torch_npu.profiler.ExportType.Db
        if export_type == "db"
        else torch_npu.profiler.ExportType.Text
    )
    return torch_npu.profiler._ExperimentalConfig(
        export_type=profiler_export_type,
        profiler_level=profiler_level,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )


def profile_operator(
    torch: Any,
    torch_npu: Any,
    case: OperatorCase,
    args: argparse.Namespace,
    raw_root: Path,
) -> None:
    state_mode = "reset" if args.reset_state else "steady"
    trace_name = (
        f"asu_hbm_index_{args.op}_bs{args.batch_size}_{state_mode}"
    )
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(raw_root),
        worker_name=trace_name,
        analyse_flag=True,
        async_mode=False,
    )
    experimental_config = create_experimental_config(
        torch_npu, args.op, args.export_type
    )

    retained_outputs = []
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        experimental_config=experimental_config,
        on_trace_ready=handler,
    ):
        for _ in range(args.profile_iterations):
            if args.reset_state:
                case.reset()
                torch.npu.synchronize()
            if args.op == "lookup":
                retained_outputs.append(
                    invoke_lookup(torch, case, args.batch_size)
                )
            else:
                invoke_maintain(
                    torch, case, args.batch_size, args.seed
                )
            if args.reset_state:
                torch.npu.synchronize()
        torch.npu.synchronize()

    if args.op == "lookup":
        if len(retained_outputs) != args.profile_iterations:
            raise RuntimeError("lookup profiling did not retain every output")
        validate_lookup_output(retained_outputs[-1], args.batch_size)


def is_raw_profile_dir(path: Path) -> bool:
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    return any(child.is_dir() and child.name == "FRAMEWORK" for child in children) or any(
        child.is_dir() and child.name.startswith("PROF_") for child in children
    )


def discover_raw_profile(raw_root: Path) -> Path:
    profiles = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_dir() and is_raw_profile_dir(path)
    )
    if len(profiles) != 1:
        rendered = ", ".join(str(path) for path in profiles) or "none"
        raise RuntimeError(
            "expected exactly one parsed raw profile directory under "
            f"{raw_root}, found {len(profiles)}: {rendered}"
        )
    return profiles[0]


def copy_parsed_results(
    raw_profile: Path,
    parsed_root: Path,
    export_type: str,
) -> list[Path]:
    profiler_output = raw_profile / "ASCEND_PROFILER_OUTPUT"
    if not profiler_output.is_dir():
        raise RuntimeError(
            "torch-npu did not create parsed ASCEND_PROFILER_OUTPUT: "
            f"{profiler_output}"
        )

    output_files = sorted(
        path for path in profiler_output.rglob("*") if path.is_file()
    )
    expected_suffixes = {".db"} if export_type == "db" else {".csv", ".json"}
    expected_outputs = [
        path for path in output_files if path.suffix.lower() in expected_suffixes
    ]
    if not expected_outputs:
        suffixes = ", ".join(sorted(expected_suffixes))
        raise RuntimeError(
            f"torch-npu parsing produced no {suffixes} output under "
            f"{profiler_output}"
        )

    copied: list[Path] = []
    parsed_output = parsed_root / "ASCEND_PROFILER_OUTPUT"
    for source in output_files:
        destination = parsed_output / source.relative_to(profiler_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    for pattern in ("profiler_info*.json", "profiler_metadata.json"):
        for source in raw_profile.glob(pattern):
            destination = parsed_root / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
    return sorted(copied)


def git_commit(package_dir: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=package_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def device_name(torch: Any, device_id: int) -> str | None:
    try:
        return str(torch.npu.get_device_name(device_id))
    except (AttributeError, RuntimeError):
        return None


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    torch: Any,
    torch_npu: Any,
    package_dir: Path,
    extension_path: Path,
    aicpu_opp: Path,
    raw_profile: Path,
    parsed_files: list[Path],
) -> None:
    profiler_level = "Level1" if args.op == "lookup" else "Level2"
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operator": (
            "_C_ascend::asu_hbm_index_lookup"
            if args.op == "lookup"
            else "_C_ascend::asu_hbm_index_maintain_aicpu"
        ),
        "configuration": {
            "batch_size": args.batch_size,
            "query_count_per_request": QUERY_COUNT,
            "fixed_updates_or_evictions_per_request": FIXED_UPDATE_COUNT,
            "warmup_iterations": args.warmup_iterations,
            "profile_iterations": args.profile_iterations,
            "state_mode": "reset" if args.reset_state else "steady",
            "seed": args.seed,
            "device_id": args.device_id,
            "profiler_level": profiler_level,
            "aic_metrics": "PipeUtilization",
            "export_type": args.export_type,
            "direct_parse": True,
        },
        "environment": {
            "device_name": device_name(torch, args.device_id),
            "torch_version": getattr(torch, "__version__", None),
            "torch_npu_version": getattr(torch_npu, "__version__", None),
            "vllm_ascend_package": str(package_dir),
            "vllm_ascend_extension": str(extension_path),
            "aicpu_opp": str(aicpu_opp),
            "git_commit": git_commit(package_dir),
        },
        "outputs": {
            "raw_profile": str(raw_profile),
            "parsed_files": [str(output) for output in parsed_files],
        },
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_root, raw_root, parsed_root = prepare_output_dir(args.output_dir)
    package_dir, aicpu_opp = configure_custom_opp()

    import torch
    import torch_npu

    extension_path = load_custom_ops(torch)
    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)
    torch.set_grad_enabled(False)

    print(f"[INFO] operator={args.op}")
    print(f"[INFO] device={device}")
    print(f"[INFO] batch_size={args.batch_size}")
    print(
        f"[INFO] warmup={args.warmup_iterations}, "
        f"profile_iterations={args.profile_iterations}"
    )
    state_mode = "reset" if args.reset_state else "steady"
    print(f"[INFO] state_mode={state_mode}")
    print(f"[INFO] export_type={args.export_type}")
    print(f"[INFO] output={run_root}")

    case = build_case(torch, device, args.batch_size, args.reset_state)
    warmup(
        torch,
        case,
        args.op,
        args.batch_size,
        args.seed,
        args.warmup_iterations,
        args.reset_state,
    )
    print("[PASS] warmup completed")

    profile_operator(torch, torch_npu, case, args, raw_root)
    print("[PASS] profile collection and direct parsing completed")

    raw_profile = discover_raw_profile(raw_root)
    parsed_files = copy_parsed_results(
        raw_profile, parsed_root, args.export_type
    )
    write_manifest(
        run_root / "manifest.json",
        args=args,
        torch=torch,
        torch_npu=torch_npu,
        package_dir=package_dir,
        extension_path=extension_path,
        aicpu_opp=aicpu_opp,
        raw_profile=raw_profile,
        parsed_files=parsed_files,
    )

    print(f"[PASS] raw profile: {raw_profile}")
    print(f"[PASS] parsed output: {parsed_root}")
    print(f"[PASS] manifest: {run_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
