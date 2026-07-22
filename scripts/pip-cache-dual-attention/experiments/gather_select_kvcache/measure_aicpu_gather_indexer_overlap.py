#!/usr/bin/env python3
"""Measure overlap between a 100us-class AICPU mock gather and LightningIndexer."""

from __future__ import annotations

import argparse
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


def call_aicpu(mod, libs, inputs, outputs, block_size: int, mock_wait_us: int, stream: int,
               start_event: int | None = None, end_event: int | None = None,
               sync: bool = True) -> None:
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
    if start_event is not None:
        mod.check(acl.rt.record_event(start_event, stream), "acl.rt.record_event(start)")
    ret = libs.opapi.aclnnMockKVSelect(ctypes.c_void_p(), workspace_size, executor, ctypes.c_void_p(int(stream)))
    mod.check(ret, "aclnnMockKVSelect")
    if end_event is not None:
        mod.check(acl.rt.record_event(end_event, stream), "acl.rt.record_event(end)")
    if sync:
        if end_event is not None:
            mod.check(acl.rt.synchronize_event(end_event), "acl.rt.synchronize_event(end)")
        else:
            mod.check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream")


def measure_aicpu_event_ms(mod, libs, inputs, outputs, block_size: int, mock_wait_us: int,
                           stream: int, warmup: int, iters: int) -> list[float]:
    start_event, ret = acl.rt.create_event()
    mod.check(ret, "acl.rt.create_event(start)")
    end_event, ret = acl.rt.create_event()
    mod.check(ret, "acl.rt.create_event(end)")
    samples: list[float] = []
    try:
        for _ in range(warmup):
            call_aicpu(mod, libs, inputs, outputs, block_size, mock_wait_us, stream)
        for _ in range(iters):
            call_aicpu(mod, libs, inputs, outputs, block_size, mock_wait_us, stream, start_event, end_event)
            elapsed_ms, ret = acl.rt.event_elapsed_time(start_event, end_event)
            mod.check(ret, "acl.rt.event_elapsed_time")
            samples.append(float(elapsed_ms))
        return samples
    finally:
        acl.rt.destroy_event(start_event)
        acl.rt.destroy_event(end_event)


def measure_indexer_event_ms(inputs: IndexerInputs, sparse_count: int, stream: torch.npu.Stream,
                             warmup: int, iters: int) -> list[float]:
    with torch.npu.stream(stream):
        for _ in range(warmup):
            run_indexer(inputs, sparse_count)
    stream.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        with torch.npu.stream(stream):
            start.record()
            run_indexer(inputs, sparse_count)
            end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def measure_serial_wall_ms(mod, libs, aicpu_inputs, aicpu_outputs, block_size: int, acl_stream: int,
                           mock_wait_us: int,
                           indexer_inputs: IndexerInputs, sparse_count: int,
                           indexer_stream: torch.npu.Stream, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        call_aicpu(mod, libs, aicpu_inputs, aicpu_outputs, block_size, mock_wait_us, acl_stream)
        with torch.npu.stream(indexer_stream):
            run_indexer(indexer_inputs, sparse_count)
        indexer_stream.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        call_aicpu(mod, libs, aicpu_inputs, aicpu_outputs, block_size, mock_wait_us, acl_stream)
        with torch.npu.stream(indexer_stream):
            run_indexer(indexer_inputs, sparse_count)
        indexer_stream.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def measure_parallel_wall_ms(mod, libs, aicpu_inputs, aicpu_outputs, block_size: int, acl_stream: int,
                             mock_wait_us: int,
                             indexer_inputs: IndexerInputs, sparse_count: int,
                             indexer_stream: torch.npu.Stream, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        call_aicpu(mod, libs, aicpu_inputs, aicpu_outputs, block_size, mock_wait_us, acl_stream, sync=False)
        with torch.npu.stream(indexer_stream):
            run_indexer(indexer_inputs, sparse_count)
        acl.rt.synchronize_stream(acl_stream)
        indexer_stream.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        call_aicpu(mod, libs, aicpu_inputs, aicpu_outputs, block_size, mock_wait_us, acl_stream, sync=False)
        with torch.npu.stream(indexer_stream):
            run_indexer(indexer_inputs, sparse_count)
        acl.rt.synchronize_stream(acl_stream)
        indexer_stream.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure AICPU mock gather and LightningIndexer overlap.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--sparse-count", type=int, default=2048)
    parser.add_argument("--aicpu-block-size", type=int, default=1)
    parser.add_argument("--mock-wait-us", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
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
    acl_stream, ret = acl.rt.create_stream()
    mod.check(ret, "acl.rt.create_stream")
    indexer_stream = torch.npu.Stream(device=device)
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

        aicpu_ms = measure_aicpu_event_ms(
            mod, libs, aicpu_inputs, aicpu_outputs, args.aicpu_block_size, args.mock_wait_us,
            acl_stream, args.warmup, args.iters
        )
        indexer_ms = measure_indexer_event_ms(indexer_inputs, args.sparse_count, indexer_stream, args.warmup, args.iters)
        serial_ms = measure_serial_wall_ms(
            mod, libs, aicpu_inputs, aicpu_outputs, args.aicpu_block_size, acl_stream, args.mock_wait_us,
            indexer_inputs, args.sparse_count, indexer_stream, args.warmup, args.iters
        )
        parallel_ms = measure_parallel_wall_ms(
            mod, libs, aicpu_inputs, aicpu_outputs, args.aicpu_block_size, acl_stream, args.mock_wait_us,
            indexer_inputs, args.sparse_count, indexer_stream, args.warmup, args.iters
        )

        print(
            f"case bs={args.batch_size} max_seq={args.max_seq_len} sparse_count={args.sparse_count} "
            f"mock_wait_us={args.mock_wait_us} "
            f"warmup={args.warmup} iters={args.iters}"
        )
        print(f"aicpu_event_ms: {summarize(aicpu_ms)}")
        print(f"indexer_event_ms: {summarize(indexer_ms)}")
        print(f"serial_wall_ms: {summarize(serial_ms)}")
        print(f"parallel_wall_ms: {summarize(parallel_ms)}")
        avg_saved = statistics.mean(serial_ms) - statistics.mean(parallel_ms)
        overlap_eff = avg_saved / max(statistics.mean(aicpu_ms), 1e-9)
        print(f"saved_ms_avg={avg_saved:.6f} overlap_eff_vs_aicpu={overlap_eff:.3f}")
    finally:
        for tensor in aicpu_inputs + aicpu_outputs:
            tensor.destroy()
        acl.rt.destroy_stream(acl_stream)
        acl.rt.reset_device(args.device_id)
        acl.finalize()


if __name__ == "__main__":
    main()
