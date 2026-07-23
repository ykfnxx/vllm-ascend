#!/usr/bin/env python3
"""Compare one DMP Lookup or Maintain operator in eager and ACL Graph modes.

Each timed launch contains exactly one custom operator. Input construction,
state reset, correctness checks, graph capture, warmup, and optional profiler
collection are excluded from the headline NPU Event timing samples.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


INDEX_CAPACITY = 144 * 1024
TOTAL_SLOT_COUNT = 10 * 1024
RESIDENT_SLOT_COUNT = 8 * 1024
FREE_SLOT_COUNT = 2 * 1024
QUERY_SLOT_COUNT = 2 * 1024
FIXED_MISS_COUNT = 300
FIXED_HIT_COUNT = QUERY_SLOT_COUNT - FIXED_MISS_COUNT
FREE_HEAD_STRIDE = 16
NOT_FOUND = -1

DEFAULT_BATCH_SIZE = 32
DEFAULT_SEQ_LEN = 128 * 1024
DEFAULT_CAPTURE_WARMUP = 3
DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 100
DEFAULT_PROFILE_ITERATIONS = 20


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    install_opp: Path
    vendor_opp: Path
    aicpu_opp: Path
    op_api_library: Path
    extension_root: Path
    extension_libraries: tuple[Path, ...]
    source_revision: Path


@dataclass
class OperatorCase:
    baseline_state: tuple[Any, Any, Any, Any]
    mutable_state: tuple[Any, Any, Any, Any]
    req_pool_entries: Any
    query_index: Any
    seq_lens: Any
    needs_refill: Any
    last_query_slots: Any | None = None

    def reset(self) -> None:
        for tensor, baseline in zip(
            self.mutable_state, self.baseline_state, strict=True
        ):
            tensor.copy_(baseline)


@dataclass(frozen=True)
class TimingSamples:
    npu_event_us: tuple[float, ...]
    host_wall_us: tuple[float, ...]


class GraphLaunch:
    """Replay one captured graph while retaining its fixed output buffers."""

    def __init__(self, graph: Any, outputs: Any) -> None:
        self.graph = graph
        self.outputs = outputs

    def __call__(self) -> Any:
        self.graph.replay()
        return self.outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one dmp_lookup_maintain Lookup or Maintain invocation "
            "between eager launch and torch.npu.NPUGraph replay."
        )
    )
    parser.add_argument(
        "--op",
        choices=("lookup", "maintain"),
        required=True,
        help="run exactly one operator type per process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"req_num and request-pool rows (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"synthetic sequence length (default: {DEFAULT_SEQ_LEN})",
    )
    parser.add_argument(
        "--refill-mode",
        choices=("off", "on"),
        default="off",
        help=(
            "Lookup needs_refill value; on copies 10240 resident IDs per "
            "request (default: off)"
        ),
    )
    parser.add_argument(
        "--state-mode",
        choices=("steady", "reset"),
        default="steady",
        help=(
            "steady repeatedly uses the same state; reset restores the fixed "
            "tensor addresses before every launch, outside timing "
            "(default: steady)"
        ),
    )
    parser.add_argument(
        "--capture-warmup-iterations",
        type=int,
        default=DEFAULT_CAPTURE_WARMUP,
        help=(
            "eager launches on the graph stream before capture "
            f"(default: {DEFAULT_CAPTURE_WARMUP})"
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"untimed launches per mode (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"timed launches per mode (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=Path,
        help=(
            "optional empty directory for separate eager and graph "
            "torch_npu profiler traces"
        ),
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=DEFAULT_PROFILE_ITERATIONS,
        help=(
            "operator launches collected in each optional profile "
            f"(default: {DEFAULT_PROFILE_ITERATIONS})"
        ),
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="logical NPU device id (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Maintain seed attribute (default: 0, layer 0 microbatch 0)",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="dmp-lookup-maintain source/runtime root",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="skip eager/graph correctness and state equivalence checks",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional output file containing configuration and raw samples",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if not 0 < args.seq_len <= INDEX_CAPACITY:
        raise ValueError(
            f"--seq-len must be in [1, {INDEX_CAPACITY}], got {args.seq_len}"
        )
    if args.seq_len <= RESIDENT_SLOT_COUNT + FIXED_MISS_COUNT - 1:
        raise ValueError(
            "--seq-len must include all fixed synthetic query tokens; "
            f"minimum is {RESIDENT_SLOT_COUNT + FIXED_MISS_COUNT}"
        )
    if args.capture_warmup_iterations < 0:
        raise ValueError("--capture-warmup-iterations cannot be negative")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than 0")
    if args.profile_iterations <= 0:
        raise ValueError("--profile-iterations must be greater than 0")
    if args.op == "maintain" and args.refill_mode != "off":
        raise ValueError("--refill-mode only applies to --op lookup")


def prepend_env_path(name: str, paths: list[Path]) -> None:
    resolved = [str(path.resolve()) for path in paths]
    existing = [entry for entry in os.environ.get(name, "").split(os.pathsep) if entry]
    normalized = {os.path.normpath(entry) for entry in resolved}
    tail = [
        entry
        for entry in existing
        if os.path.normpath(entry) not in normalized
    ]
    os.environ[name] = os.pathsep.join([*resolved, *tail])


def configure_runtime(runtime_root: Path) -> RuntimePaths:
    root = runtime_root.expanduser().resolve()
    install_opp = Path(
        os.environ.get(
            "DMP_LOOKUP_MAINTAIN_INSTALL_OPP_PATH",
            str(root / "opp"),
        )
    ).expanduser().resolve()
    vendor_opp = install_opp / "vendors/customize"
    aicpu_opp = vendor_opp / "op_impl/aicpu_transformer"
    op_api_library = vendor_opp / "op_api/lib/libcust_opapi.so"
    extension_root = root / "torch_extension"
    extension_libraries = tuple(
        sorted(
            extension_root.glob(
                "dmp_lookup_maintain_custom_ops/*.so"
            )
        )
    )
    source_revision = root / "DMP_SOURCE_REVISION.txt"

    missing = [
        path
        for path in (
            vendor_opp,
            aicpu_opp,
            op_api_library,
            extension_root,
        )
        if not path.exists()
    ]
    if not extension_libraries:
        missing.append(
            extension_root / "dmp_lookup_maintain_custom_ops/*.so"
        )
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            "DMP Lookup/Maintain runtime is incomplete:\n"
            f"{details}\n"
            "Run scripts/build_dmp_lookup_maintain.sh first."
        )

    os.environ["DMP_LOOKUP_MAINTAIN_INSTALL_OPP_PATH"] = str(install_opp)
    prepend_env_path("ASCEND_CUSTOM_OPP_PATH", [aicpu_opp, vendor_opp])
    prepend_env_path("LD_LIBRARY_PATH", [op_api_library.parent])
    sys.path.insert(0, str(extension_root))

    return RuntimePaths(
        root=root,
        install_opp=install_opp,
        vendor_opp=vendor_opp,
        aicpu_opp=aicpu_opp,
        op_api_library=op_api_library,
        extension_root=extension_root,
        extension_libraries=extension_libraries,
        source_revision=source_revision,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_custom_ops(torch: Any) -> Any:
    extension = importlib.import_module("dmp_lookup_maintain_custom_ops")
    for op_name in (
        "asu_hbm_index_lookup",
        "asu_hbm_index_maintain_aicpu",
    ):
        qualified_name = f"dmp_lookup_maintain::{op_name}"
        if not hasattr(torch.ops.dmp_lookup_maintain, op_name):
            raise RuntimeError(f"PyTorch operator is not registered: {qualified_name}")
        for dispatch_key in ("PrivateUse1", "Meta"):
            if not torch._C._dispatch_has_kernel_for_dispatch_key(
                qualified_name, dispatch_key
            ):
                raise RuntimeError(
                    f"{qualified_name} has no {dispatch_key} implementation"
                )
    return extension


def build_case(
    torch: Any,
    device: Any,
    batch_size: int,
    seq_len: int,
    refill_mode: str,
) -> OperatorCase:
    initial_tokens = torch.arange(RESIDENT_SLOT_COUNT, dtype=torch.int32)
    token_to_slot = torch.full(
        (batch_size, INDEX_CAPACITY), NOT_FOUND, dtype=torch.int32
    )
    slot_to_token = torch.full(
        (batch_size, TOTAL_SLOT_COUNT), NOT_FOUND, dtype=torch.int32
    )
    token_to_slot[:, :RESIDENT_SLOT_COUNT] = initial_tokens
    slot_to_token[:, :RESIDENT_SLOT_COUNT] = initial_tokens
    free_slots = (
        torch.arange(
            RESIDENT_SLOT_COUNT,
            TOTAL_SLOT_COUNT,
            dtype=torch.int32,
        )
        .view(1, -1)
        .expand(batch_size, -1)
        .clone()
    )
    free_head = torch.zeros(
        (batch_size, FREE_HEAD_STRIDE), dtype=torch.int32
    )
    req_pool_entries = torch.arange(batch_size, dtype=torch.int32)
    hits = torch.arange(FIXED_HIT_COUNT, dtype=torch.int32)
    misses = torch.arange(
        RESIDENT_SLOT_COUNT,
        RESIDENT_SLOT_COUNT + FIXED_MISS_COUNT,
        dtype=torch.int32,
    )
    query_index = (
        torch.cat((hits, misses))
        .view(1, -1)
        .expand(batch_size, -1)
        .clone()
    )
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)
    needs_refill = torch.full(
        (batch_size,),
        refill_mode == "on",
        dtype=torch.bool,
    )

    baseline_state = tuple(
        tensor.to(device)
        for tensor in (
            token_to_slot,
            slot_to_token,
            free_slots,
            free_head,
        )
    )
    case = OperatorCase(
        baseline_state=baseline_state,
        mutable_state=tuple(tensor.clone() for tensor in baseline_state),
        req_pool_entries=req_pool_entries.to(device),
        query_index=query_index.to(device),
        seq_lens=seq_lens.to(device),
        needs_refill=needs_refill.to(device),
    )
    torch.npu.synchronize()
    return case


def call_lookup(torch: Any, case: OperatorCase, batch_size: int) -> tuple[Any, ...]:
    token_to_slot, slot_to_token, free_slots, free_head = case.mutable_state
    return torch.ops.dmp_lookup_maintain.asu_hbm_index_lookup(
        token_to_slot,
        slot_to_token,
        free_slots,
        free_head,
        case.req_pool_entries,
        case.query_index,
        case.seq_lens,
        case.needs_refill,
        batch_size,
    )


def call_maintain(
    torch: Any,
    case: OperatorCase,
    batch_size: int,
    seed: int,
) -> None:
    if case.last_query_slots is None:
        raise RuntimeError("Maintain case has no Lookup-produced last_query_slots")
    token_to_slot, slot_to_token, free_slots, free_head = case.mutable_state
    torch.ops.dmp_lookup_maintain.asu_hbm_index_maintain_aicpu(
        token_to_slot,
        slot_to_token,
        free_slots,
        free_head,
        case.req_pool_entries,
        case.last_query_slots,
        batch_size,
        seed,
    )


def prepare_maintain_case(
    torch: Any,
    case: OperatorCase,
    batch_size: int,
) -> None:
    """Run the producer Lookup once and save its post-Lookup state as baseline."""
    outputs = call_lookup(torch, case, batch_size)
    torch.npu.synchronize()
    case.last_query_slots = outputs[0]
    case.baseline_state = tuple(tensor.clone() for tensor in case.mutable_state)
    torch.npu.synchronize()


def make_launch(
    torch: Any,
    case: OperatorCase,
    op_name: str,
    batch_size: int,
    seed: int,
) -> Callable[[], Any]:
    if op_name == "lookup":
        return lambda: call_lookup(torch, case, batch_size)
    return lambda: call_maintain(torch, case, batch_size, seed)


def reset_case(torch: Any, case: OperatorCase, stream: Any) -> None:
    with torch.npu.stream(stream):
        case.reset()
    stream.synchronize()


def capture_graph(
    torch: Any,
    case: OperatorCase,
    stream: Any,
    launch: Callable[[], Any],
    capture_warmup_iterations: int,
) -> GraphLaunch:
    for _ in range(capture_warmup_iterations):
        reset_case(torch, case, stream)
        with torch.npu.stream(stream):
            outputs = launch()
        stream.synchronize()
        del outputs

    reset_case(torch, case, stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        with torch.npu.graph(graph):
            graph_outputs = launch()
    stream.synchronize()
    return GraphLaunch(graph, graph_outputs)


def execute_once(
    torch: Any,
    case: OperatorCase,
    stream: Any,
    launch: Callable[[], Any],
) -> Any:
    reset_case(torch, case, stream)
    with torch.npu.stream(stream):
        outputs = launch()
    stream.synchronize()
    return outputs


def assert_equal(torch: Any, name: str, actual: Any, expected: Any) -> None:
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    if torch.equal(actual_cpu, expected_cpu):
        return
    mismatch = (actual_cpu != expected_cpu).nonzero()
    first = tuple(int(value) for value in mismatch[0])
    raise AssertionError(
        f"{name} mismatch at {first}: "
        f"actual={actual_cpu[first].item()}, "
        f"expected={expected_cpu[first].item()}, "
        f"total_mismatches={int(mismatch.shape[0])}"
    )


def validate_lookup_expected(
    torch: Any,
    case: OperatorCase,
    outputs: tuple[Any, ...],
    batch_size: int,
    refill_mode: str,
    prefix: str,
) -> None:
    slot_out, miss_out, hit_sparse, miss_sparse, resident_ids = outputs
    hit_slots = torch.arange(FIXED_HIT_COUNT, dtype=torch.int32)
    miss_slots = torch.arange(
        RESIDENT_SLOT_COUNT,
        RESIDENT_SLOT_COUNT + FIXED_MISS_COUNT,
        dtype=torch.int32,
    )
    expected_slots = (
        torch.cat((hit_slots, miss_slots))
        .view(1, -1)
        .expand(batch_size, -1)
    )
    expected_misses = torch.zeros(
        (batch_size, QUERY_SLOT_COUNT), dtype=torch.int32
    )
    expected_misses[:, FIXED_HIT_COUNT:] = 1
    expected_hit_sparse = torch.full_like(expected_misses, NOT_FOUND)
    expected_hit_sparse[:, :FIXED_HIT_COUNT] = hit_slots
    expected_miss_sparse = torch.full_like(expected_misses, NOT_FOUND)
    expected_miss_sparse[:, FIXED_HIT_COUNT:] = miss_slots

    assert_equal(torch, f"{prefix} slot_out", slot_out, expected_slots)
    assert_equal(torch, f"{prefix} miss_out", miss_out, expected_misses)
    assert_equal(
        torch,
        f"{prefix} hit_sparse_indices",
        hit_sparse,
        expected_hit_sparse,
    )
    assert_equal(
        torch,
        f"{prefix} miss_sparse_indices",
        miss_sparse,
        expected_miss_sparse,
    )
    if refill_mode == "on":
        assert_equal(
            torch,
            f"{prefix} resident_token_ids",
            resident_ids,
            case.mutable_state[1],
        )


def validate_modes(
    torch: Any,
    eager_case: OperatorCase,
    graph_case: OperatorCase,
    stream: Any,
    eager_launch: Callable[[], Any],
    graph_launch: Callable[[], Any],
    op_name: str,
    batch_size: int,
    refill_mode: str,
) -> None:
    eager_outputs = execute_once(torch, eager_case, stream, eager_launch)
    graph_outputs = execute_once(torch, graph_case, stream, graph_launch)

    for index, (eager_state, graph_state) in enumerate(
        zip(
            eager_case.mutable_state,
            graph_case.mutable_state,
            strict=True,
        )
    ):
        assert_equal(
            torch,
            f"{op_name} eager/graph mutable_state[{index}]",
            graph_state,
            eager_state,
        )

    if op_name == "lookup":
        validate_lookup_expected(
            torch,
            eager_case,
            eager_outputs,
            batch_size,
            refill_mode,
            "eager Lookup",
        )
        validate_lookup_expected(
            torch,
            graph_case,
            graph_outputs,
            batch_size,
            refill_mode,
            "graph Lookup",
        )
        output_count = 5 if refill_mode == "on" else 4
        for index in range(output_count):
            assert_equal(
                torch,
                f"Lookup eager/graph output[{index}]",
                graph_outputs[index],
                eager_outputs[index],
            )


def measure_mode(
    torch: Any,
    case: OperatorCase,
    stream: Any,
    launch: Callable[[], Any],
    state_mode: str,
    warmup_iterations: int,
    iterations: int,
) -> TimingSamples:
    reset_case(torch, case, stream)
    for _ in range(warmup_iterations):
        if state_mode == "reset":
            reset_case(torch, case, stream)
        with torch.npu.stream(stream):
            outputs = launch()
        stream.synchronize()
        del outputs

    npu_event_us: list[float] = []
    host_wall_us: list[float] = []
    for _ in range(iterations):
        if state_mode == "reset":
            reset_case(torch, case, stream)

        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        host_start_ns = time.perf_counter_ns()
        with torch.npu.stream(stream):
            start.record()
            outputs = launch()
            end.record()
        end.synchronize()
        host_end_ns = time.perf_counter_ns()

        npu_event_us.append(float(start.elapsed_time(end)) * 1000.0)
        host_wall_us.append((host_end_ns - host_start_ns) / 1000.0)
        del outputs

    return TimingSamples(
        npu_event_us=tuple(npu_event_us),
        host_wall_us=tuple(host_wall_us),
    )


def percentile(samples: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def summarize_samples(samples: tuple[float, ...]) -> dict[str, Any]:
    return {
        "count": len(samples),
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "p95_us": percentile(samples, 0.95),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": list(samples),
    }


def summarize_mode(mode: str, samples: TimingSamples) -> dict[str, Any]:
    return {
        "mode": mode,
        "npu_event": summarize_samples(samples.npu_event_us),
        "host_wall": summarize_samples(samples.host_wall_us),
    }


def print_summary(op_name: str, mode: str, result: dict[str, Any]) -> None:
    npu = result["npu_event"]
    host = result["host_wall"]
    print(
        f"[RESULT] op={op_name} mode={mode} "
        f"average={npu['mean_us']:.3f}us/op "
        f"median={npu['median_us']:.3f}us/op "
        f"p95={npu['p95_us']:.3f}us/op "
        f"min={npu['min_us']:.3f}us/op "
        f"max={npu['max_us']:.3f}us/op "
        f"host_average={host['mean_us']:.3f}us/op"
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
    case: OperatorCase,
    stream: Any,
    launch: Callable[[], Any],
    state_mode: str,
    iterations: int,
    output_dir: Path,
    op_name: str,
    mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(output_dir),
        worker_name=f"dmp_{op_name}_{mode}",
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

    reset_case(torch, case, stream)
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
            if state_mode == "reset":
                reset_case(torch, case, stream)
            with torch.npu.stream(stream):
                outputs = launch()
            stream.synchronize()
            del outputs


def print_runtime_info(
    paths: RuntimePaths,
    extension: Any,
    device: Any,
    args: argparse.Namespace,
) -> None:
    print(f"[INFO] device={device}")
    print(f"[INFO] extension_module={Path(extension.__file__).resolve()}")
    print(f"[INFO] op_api_library={paths.op_api_library}")
    print(
        f"[INFO] op_api_sha256={sha256_file(paths.op_api_library)}"
    )
    for extension_library in paths.extension_libraries:
        print(f"[INFO] extension_library={extension_library}")
        print(
            f"[INFO] extension_sha256={sha256_file(extension_library)}"
        )
    if paths.source_revision.is_file():
        for line in paths.source_revision.read_text(
            encoding="utf-8"
        ).splitlines():
            print(f"[INFO] source_revision={line}")
    print(
        f"[INFO] op={args.op} batch_size={args.batch_size} "
        f"seq_len={args.seq_len} fixed_misses={FIXED_MISS_COUNT} "
        f"refill_mode={args.refill_mode} state_mode={args.state_mode} "
        f"seed={args.seed}"
    )
    print(
        f"[INFO] capture_warmup={args.capture_warmup_iterations} "
        f"warmup={args.warmup_iterations} iterations={args.iterations}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    runtime_paths = configure_runtime(args.runtime_root)

    import torch
    import torch_npu

    extension = load_custom_ops(torch)
    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)
    torch.set_grad_enabled(False)
    stream = torch.npu.Stream(device=device)

    eager_case = build_case(
        torch,
        device,
        args.batch_size,
        args.seq_len,
        args.refill_mode,
    )
    graph_case = build_case(
        torch,
        device,
        args.batch_size,
        args.seq_len,
        args.refill_mode,
    )
    if args.op == "maintain":
        prepare_maintain_case(torch, eager_case, args.batch_size)
        prepare_maintain_case(torch, graph_case, args.batch_size)

    eager_launch = make_launch(
        torch,
        eager_case,
        args.op,
        args.batch_size,
        args.seed,
    )
    graph_capture_launch = make_launch(
        torch,
        graph_case,
        args.op,
        args.batch_size,
        args.seed,
    )
    graph_launch = capture_graph(
        torch,
        graph_case,
        stream,
        graph_capture_launch,
        args.capture_warmup_iterations,
    )

    if not args.skip_check:
        validate_modes(
            torch,
            eager_case,
            graph_case,
            stream,
            eager_launch,
            graph_launch,
            args.op,
            args.batch_size,
            args.refill_mode,
        )
        print("[PASS] eager and graph outputs/state are equivalent")

    print_runtime_info(runtime_paths, extension, device, args)

    eager_samples = measure_mode(
        torch,
        eager_case,
        stream,
        eager_launch,
        args.state_mode,
        args.warmup_iterations,
        args.iterations,
    )
    graph_samples = measure_mode(
        torch,
        graph_case,
        stream,
        graph_launch,
        args.state_mode,
        args.warmup_iterations,
        args.iterations,
    )
    eager_result = summarize_mode("eager", eager_samples)
    graph_result = summarize_mode("graph", graph_samples)
    print_summary(args.op, "eager", eager_result)
    print_summary(args.op, "graph", graph_result)

    eager_average = eager_result["npu_event"]["mean_us"]
    graph_average = graph_result["npu_event"]["mean_us"]
    graph_to_eager = graph_average / eager_average
    print(
        f"[AVERAGE] eager_{args.op}={eager_average:.3f}us/op "
        f"graph_{args.op}={graph_average:.3f}us/op "
        f"graph/eager={graph_to_eager:.3f}x"
    )

    profile_root = None
    if args.profile_output_dir is not None:
        profile_root = prepare_profile_root(args.profile_output_dir)
        profile_mode(
            torch,
            torch_npu,
            eager_case,
            stream,
            eager_launch,
            args.state_mode,
            args.profile_iterations,
            profile_root / f"{args.op}_eager",
            args.op,
            "eager",
        )
        profile_mode(
            torch,
            torch_npu,
            graph_case,
            stream,
            graph_launch,
            args.state_mode,
            args.profile_iterations,
            profile_root / f"{args.op}_graph",
            args.op,
            "graph",
        )
        print(
            f"[PASS] wrote {args.profile_iterations} eager and graph profile "
            f"iterations to {profile_root}"
        )

    if args.output_json is not None:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "op": args.op,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "fixed_miss_count": FIXED_MISS_COUNT,
                "refill_mode": args.refill_mode,
                "state_mode": args.state_mode,
                "seed": args.seed,
                "device": str(device),
                "capture_warmup_iterations": args.capture_warmup_iterations,
                "warmup_iterations": args.warmup_iterations,
                "iterations": args.iterations,
                "profile_iterations": args.profile_iterations,
                "profile_output_dir": (
                    str(profile_root) if profile_root is not None else None
                ),
            },
            "runtime": {
                "root": str(runtime_paths.root),
                "op_api_library": str(runtime_paths.op_api_library),
                "op_api_sha256": sha256_file(runtime_paths.op_api_library),
                "extension_module": str(Path(extension.__file__).resolve()),
                "extension_libraries": [
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                    for path in runtime_paths.extension_libraries
                ],
            },
            "eager": eager_result,
            "graph": graph_result,
            "graph_to_eager_mean_ratio": graph_to_eager,
        }
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[PASS] wrote JSON report to {output_path}")


if __name__ == "__main__":
    main()
