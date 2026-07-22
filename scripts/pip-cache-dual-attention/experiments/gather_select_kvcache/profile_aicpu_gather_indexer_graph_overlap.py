#!/usr/bin/env python3
"""Measure and profile graph replay overlap between AICPU mock gather and LightningIndexer."""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib.util
import statistics
import sys
import time
from pathlib import Path
from typing import NamedTuple

import acl


REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_ACL_TEST = REPO_ROOT / "op/aicpu_mock_kv_select/tests/test_mock_kv_select_acl.py"
DEFAULT_TRACE_ROOT = Path(__file__).resolve().parent / "profiler_trace/aicpu_gather_indexer_graph_overlap"

INDEXER_BLOCK_SIZE = 128
INDEXER_HEADS = 64
INDEXER_HEAD_DIM = 128
INDEXER_SPARSE_MODE = 3


class IndexerInputs(NamedTuple):
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    q_lens: torch.Tensor
    k_lens: torch.Tensor
    block_table: torch.Tensor


class PreparedAicpuCall(NamedTuple):
    workspace_size: int
    executor: ctypes.c_void_p


def load_mock_acl_module():
    module_name = "mock_kv_select_acl"
    spec = importlib.util.spec_from_file_location(module_name, MOCK_ACL_TEST)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct / 100.0))
    return ordered[index]


def summarize(values: list[float]) -> str:
    return (
        f"avg={statistics.mean(values):.6f} "
        f"p50={statistics.median(values):.6f} "
        f"p90={percentile(values, 90):.6f} "
        f"min={min(values):.6f} max={max(values):.6f}"
    )


def make_indexer_inputs(batch_size: int, key_seq_len: int, sparse_count: int, device: torch.device) -> IndexerInputs:
    del sparse_count
    blocks_per_seq = (key_seq_len + INDEXER_BLOCK_SIZE - 1) // INDEXER_BLOCK_SIZE
    num_key_blocks = batch_size * blocks_per_seq
    token_count = batch_size
    query = torch.randn(token_count, INDEXER_HEADS, INDEXER_HEAD_DIM, dtype=INDEXER_DTYPE, device=device)
    key = torch.randn(num_key_blocks, INDEXER_BLOCK_SIZE, 1, INDEXER_HEAD_DIM, dtype=INDEXER_DTYPE, device=device)
    weights = torch.randn(token_count, INDEXER_HEADS, dtype=INDEXER_DTYPE, device=device)
    q_lens = torch.cumsum(torch.ones(batch_size, dtype=torch.int32, device=device), dim=0).to(torch.int32)
    k_lens = torch.full((batch_size,), key_seq_len, dtype=torch.int32, device=device)
    block_table = torch.arange(num_key_blocks, dtype=torch.int32, device=device).view(batch_size, blocks_per_seq)
    return IndexerInputs(query, key, weights, q_lens, k_lens, block_table)


def run_indexer(inputs: IndexerInputs, sparse_count: int) -> torch.Tensor:
    out = torch_npu.npu_lightning_indexer(
        query=inputs.query,
        key=inputs.key,
        weights=inputs.weights,
        actual_seq_lengths_query=inputs.q_lens,
        actual_seq_lengths_key=inputs.k_lens,
        block_table=inputs.block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        sparse_mode=INDEXER_SPARSE_MODE,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


def prepare_aicpu_call(mod, libs, inputs, outputs, block_size: int, mock_wait_us: int) -> PreparedAicpuCall:
    workspace_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p()
    ret = libs.opapi.aclnnMockKVSelectGetWorkspaceSize(
        *[ctypes.c_void_p(tensor.tensor) for tensor in inputs],
        ctypes.c_int64(block_size),
        ctypes.c_int64(mock_wait_us),
        *[ctypes.c_void_p(tensor.tensor) for tensor in outputs],
        ctypes.byref(workspace_size),
        ctypes.byref(executor),
    )
    mod.check(ret, "aclnnMockKVSelectGetWorkspaceSize")
    return PreparedAicpuCall(workspace_size.value, executor)


def launch_aicpu(mod, libs, prepared: PreparedAicpuCall, stream: torch.npu.Stream) -> None:
    ret = libs.opapi.aclnnMockKVSelect(
        ctypes.c_void_p(),
        ctypes.c_uint64(prepared.workspace_size),
        prepared.executor,
        ctypes.c_void_p(int(stream.npu_stream)),
    )
    mod.check(ret, "aclnnMockKVSelect")


def launch_fresh_aicpu(mod, libs, inputs, outputs, block_size: int, mock_wait_us: int,
                       stream: torch.npu.Stream) -> None:
    prepared = prepare_aicpu_call(mod, libs, inputs, outputs, block_size, mock_wait_us)
    launch_aicpu(mod, libs, prepared, stream)


def capture_aicpu_graph(mod, libs, inputs, outputs, block_size: int, mock_wait_us: int,
                        stream: torch.npu.Stream, warmup: int) -> torch.npu.NPUGraph:
    with torch.npu.stream(stream):
        for _ in range(warmup):
            launch_fresh_aicpu(mod, libs, inputs, outputs, block_size, mock_wait_us, stream)
    stream.synchronize()

    graph = torch.npu.NPUGraph()
    prepared = prepare_aicpu_call(mod, libs, inputs, outputs, block_size, mock_wait_us)
    with torch.npu.graph(graph, stream=stream, capture_error_mode="relaxed"):
        launch_aicpu(mod, libs, prepared, stream)
    return graph


def capture_indexer_graph(inputs: IndexerInputs, sparse_count: int, stream: torch.npu.Stream,
                          warmup: int) -> torch.npu.NPUGraph:
    with torch.npu.stream(stream):
        for _ in range(warmup):
            run_indexer(inputs, sparse_count)
    stream.synchronize()

    graph = torch.npu.NPUGraph()
    holder = []
    with torch.npu.graph(graph, stream=stream, capture_error_mode="relaxed"):
        holder.append(run_indexer(inputs, sparse_count))
    return graph


def capture_serial_graph(mod, libs, aicpu_inputs, aicpu_outputs, block_size: int,
                         mock_wait_us: int,
                         indexer_inputs: IndexerInputs, sparse_count: int,
                         stream: torch.npu.Stream, warmup: int) -> torch.npu.NPUGraph:
    with torch.npu.stream(stream):
        for _ in range(warmup):
            launch_fresh_aicpu(mod, libs, aicpu_inputs, aicpu_outputs, block_size, mock_wait_us, stream)
            run_indexer(indexer_inputs, sparse_count)
    stream.synchronize()

    graph = torch.npu.NPUGraph()
    holder = []
    prepared = prepare_aicpu_call(mod, libs, aicpu_inputs, aicpu_outputs, block_size, mock_wait_us)
    with torch.npu.graph(graph, stream=stream, capture_error_mode="relaxed"):
        launch_aicpu(mod, libs, prepared, stream)
        holder.append(run_indexer(indexer_inputs, sparse_count))
    return graph


def measure_graph_event_ms(graph: torch.npu.NPUGraph, stream: torch.npu.Stream,
                           warmup: int, iters: int) -> list[float]:
    with torch.npu.stream(stream):
        for _ in range(warmup):
            graph.replay()
    stream.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        with torch.npu.stream(stream):
            start.record()
            graph.replay()
            end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def replay_parallel_once(aicpu_graph: torch.npu.NPUGraph, aicpu_stream: torch.npu.Stream,
                         indexer_graph: torch.npu.NPUGraph, indexer_stream: torch.npu.Stream,
                         order: str) -> None:
    if order == "aicpu_first":
        with torch.npu.stream(aicpu_stream):
            aicpu_graph.replay()
        with torch.npu.stream(indexer_stream):
            indexer_graph.replay()
    elif order == "indexer_first":
        with torch.npu.stream(indexer_stream):
            indexer_graph.replay()
        with torch.npu.stream(aicpu_stream):
            aicpu_graph.replay()
    else:
        raise ValueError(f"unsupported parallel replay order: {order}")


def measure_parallel_wall_ms(aicpu_graph: torch.npu.NPUGraph, aicpu_stream: torch.npu.Stream,
                             indexer_graph: torch.npu.NPUGraph, indexer_stream: torch.npu.Stream,
                             order: str, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        replay_parallel_once(aicpu_graph, aicpu_stream, indexer_graph, indexer_stream, order)
        aicpu_stream.synchronize()
        indexer_stream.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        replay_parallel_once(aicpu_graph, aicpu_stream, indexer_graph, indexer_stream, order)
        aicpu_stream.synchronize()
        indexer_stream.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def measure_parallel_event_ms(aicpu_graph: torch.npu.NPUGraph, aicpu_stream: torch.npu.Stream,
                              indexer_graph: torch.npu.NPUGraph, indexer_stream: torch.npu.Stream,
                              order: str, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        replay_parallel_once(aicpu_graph, aicpu_stream, indexer_graph, indexer_stream, order)
        aicpu_stream.synchronize()
        indexer_stream.synchronize()

    samples: list[float] = []
    main_stream = torch.npu.current_stream()
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        done_aicpu = torch.npu.Event()
        done_indexer = torch.npu.Event()

        start.record(main_stream)
        aicpu_stream.wait_event(start)
        indexer_stream.wait_event(start)
        if order == "aicpu_first":
            with torch.npu.stream(aicpu_stream):
                aicpu_graph.replay()
                done_aicpu.record()
            with torch.npu.stream(indexer_stream):
                indexer_graph.replay()
                done_indexer.record()
        elif order == "indexer_first":
            with torch.npu.stream(indexer_stream):
                indexer_graph.replay()
                done_indexer.record()
            with torch.npu.stream(aicpu_stream):
                aicpu_graph.replay()
                done_aicpu.record()
        else:
            raise ValueError(f"unsupported parallel replay order: {order}")
        main_stream.wait_event(done_aicpu)
        main_stream.wait_event(done_indexer)
        end.record(main_stream)
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def measure_serial_graph_wall_ms(serial_graph: torch.npu.NPUGraph, stream: torch.npu.Stream,
                                 warmup: int, iters: int) -> list[float]:
    with torch.npu.stream(stream):
        for _ in range(warmup):
            serial_graph.replay()
    stream.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        with torch.npu.stream(stream):
            serial_graph.replay()
        stream.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def make_profiler(trace_dir: Path, warmup: int, active: int):
    trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ]
    experimental_config = None
    try:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            export_type=torch_npu.profiler.ExportType.Text,
        )
    except Exception:
        pass

    return torch_npu.profiler.profile(
        activities=activities,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir), analyse_flag=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental_config,
    )


def profile_serial_graph(serial_graph: torch.npu.NPUGraph, stream: torch.npu.Stream,
                         trace_dir: Path, warmup: int, active: int) -> None:
    with make_profiler(trace_dir, warmup, active) as prof:
        for _ in range(warmup + active):
            with torch.npu.stream(stream):
                serial_graph.replay()
            stream.synchronize()
            prof.step()


def profile_parallel_graphs(aicpu_graph: torch.npu.NPUGraph, aicpu_stream: torch.npu.Stream,
                            indexer_graph: torch.npu.NPUGraph, indexer_stream: torch.npu.Stream,
                            trace_dir: Path, order: str, warmup: int, active: int) -> None:
    with make_profiler(trace_dir, warmup, active) as prof:
        for _ in range(warmup + active):
            replay_parallel_once(aicpu_graph, aicpu_stream, indexer_graph, indexer_stream, order)
            aicpu_stream.synchronize()
            indexer_stream.synchronize()
            prof.step()


def parse_time_us(row: dict[str, str]) -> float:
    for key in ("Total Time(us)", "total_time_us", "Total Time (us)", "Duration(us)", "duration_us"):
        value = row.get(key)
        if value:
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def parse_name(row: dict[str, str]) -> str:
    for key in ("OP Type", "Op Name", "OP Name", "Name", "op_name", "OpName"):
        value = row.get(key)
        if value:
            return value
    return "?"


def summarize_trace_dir(trace_dir: Path) -> dict:
    out = {"trace_dir": str(trace_dir), "op_csv": "", "top_ops": []}
    csv_candidates = list(trace_dir.rglob("op_statistic.csv"))
    if not csv_candidates:
        csv_candidates = list(trace_dir.rglob("op_summary*.csv"))
    if not csv_candidates:
        return out

    csv_path = max(csv_candidates, key=lambda p: p.stat().st_mtime)
    out["op_csv"] = str(csv_path)
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    ranked = sorted(rows, key=parse_time_us, reverse=True)[:12]
    out["top_ops"] = [{"name": parse_name(row), "total_us": parse_time_us(row)} for row in ranked]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile graph replay overlap for AICPU mock gather and LightningIndexer.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--sparse-count", type=int, default=2048)
    parser.add_argument("--aicpu-block-size", type=int, default=1)
    parser.add_argument("--mock-wait-us", type=int, default=25)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument("--profile-active", type=int, default=5)
    parser.add_argument("--parallel-order", choices=("aicpu_first", "indexer_first"), default="aicpu_first")
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mod = load_mock_acl_module()
    mod.check(acl.init(), "acl.init")
    mod.check(acl.rt.set_device(args.device_id), "acl.rt.set_device")

    global torch, torch_npu, INDEXER_DTYPE
    import torch
    import torch_npu

    INDEXER_DTYPE = torch.bfloat16
    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)

    libs = mod.AclnnLibs(mod.default_opapi_path())
    aicpu_inputs = []
    aicpu_outputs = []
    try:
        input_specs, output_specs = mod.make_specs(
            args.batch_size, 1, 1, 16, args.max_seq_len, args.sparse_count, args.aicpu_block_size
        )
        aicpu_inputs = [mod.AclTensor.create(libs, dtype, shape) for dtype, shape in input_specs]
        aicpu_outputs = [mod.AclTensor.create(libs, dtype, shape) for dtype, shape in output_specs]
        indexer_inputs = make_indexer_inputs(args.batch_size, args.max_seq_len, args.sparse_count, device)

        aicpu_stream = torch.npu.Stream(device=device)
        indexer_stream = torch.npu.Stream(device=device)
        serial_stream = torch.npu.Stream(device=device)

        aicpu_graph = capture_aicpu_graph(
            mod, libs, aicpu_inputs, aicpu_outputs, args.aicpu_block_size, args.mock_wait_us,
            aicpu_stream, args.capture_warmup
        )
        indexer_graph = capture_indexer_graph(indexer_inputs, args.sparse_count, indexer_stream, args.capture_warmup)
        serial_graph = capture_serial_graph(
            mod, libs, aicpu_inputs, aicpu_outputs, args.aicpu_block_size, args.mock_wait_us,
            indexer_inputs, args.sparse_count, serial_stream, args.capture_warmup
        )

        aicpu_graph_ms = measure_graph_event_ms(aicpu_graph, aicpu_stream, args.warmup, args.iters)
        indexer_graph_ms = measure_graph_event_ms(indexer_graph, indexer_stream, args.warmup, args.iters)
        serial_graph_event_ms = measure_graph_event_ms(serial_graph, serial_stream, args.warmup, args.iters)
        serial_graph_wall_ms = measure_serial_graph_wall_ms(serial_graph, serial_stream, args.warmup, args.iters)
        parallel_event_ms = measure_parallel_event_ms(
            aicpu_graph, aicpu_stream, indexer_graph, indexer_stream, args.parallel_order, args.warmup, args.iters
        )
        parallel_wall_ms = measure_parallel_wall_ms(
            aicpu_graph, aicpu_stream, indexer_graph, indexer_stream, args.parallel_order, args.warmup, args.iters
        )

        avg_saved = statistics.mean(serial_graph_wall_ms) - statistics.mean(parallel_wall_ms)
        device_saved = statistics.mean(serial_graph_event_ms) - statistics.mean(parallel_event_ms)
        overlap_eff = device_saved / max(statistics.mean(aicpu_graph_ms), 1e-9)

        print(
            f"case bs={args.batch_size} max_seq={args.max_seq_len} sparse_count={args.sparse_count} "
            f"mock_wait_us={args.mock_wait_us} parallel_order={args.parallel_order} capture_warmup={args.capture_warmup} "
            f"warmup={args.warmup} iters={args.iters}"
        )
        print(f"aicpu_graph_event_ms: {summarize(aicpu_graph_ms)}")
        print(f"indexer_graph_event_ms: {summarize(indexer_graph_ms)}")
        print(f"serial_graph_event_ms: {summarize(serial_graph_event_ms)}")
        print(f"parallel_graph_event_ms: {summarize(parallel_event_ms)}")
        print(f"serial_graph_wall_ms: {summarize(serial_graph_wall_ms)}")
        print(f"parallel_graph_wall_ms: {summarize(parallel_wall_ms)}")
        print(
            f"device_saved_ms_avg={device_saved:.6f} wall_saved_ms_avg={avg_saved:.6f} "
            f"overlap_eff_vs_aicpu_graph={overlap_eff:.3f}"
        )

        if not args.skip_profile:
            tag = f"bs{args.batch_size}_seq{args.max_seq_len}_topk{args.sparse_count}_wait{args.mock_wait_us}"
            serial_trace = args.trace_root / tag / "serial_graph"
            parallel_trace = args.trace_root / tag / f"parallel_graph_{args.parallel_order}"
            profile_serial_graph(serial_graph, serial_stream, serial_trace, args.profile_warmup, args.profile_active)
            profile_parallel_graphs(
                aicpu_graph, aicpu_stream, indexer_graph, indexer_stream,
                parallel_trace, args.parallel_order, args.profile_warmup, args.profile_active
            )
            print(f"serial_profile: {summarize_trace_dir(serial_trace)}")
            print(f"parallel_profile: {summarize_trace_dir(parallel_trace)}")
    finally:
        for tensor in aicpu_inputs + aicpu_outputs:
            tensor.destroy()
        acl.rt.reset_device(args.device_id)
        acl.finalize()


if __name__ == "__main__":
    main()
