#!/usr/bin/env python3
"""Measure standalone ACLNN MockKVSelect AICPU latency."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import acl


REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_ACL_TEST = REPO_ROOT / "op/aicpu_mock_kv_select/tests/test_mock_kv_select_acl.py"


def load_mock_acl_module():
    module_name = "mock_kv_select_acl_latency"
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


def summarize_us(values: list[float]) -> str:
    return (
        f"avg={statistics.mean(values):.3f} "
        f"p50={statistics.median(values):.3f} "
        f"p90={percentile(values, 90):.3f} "
        f"min={min(values):.3f} max={max(values):.3f}"
    )


def call_once(mod, libs, inputs, outputs, block_size: int, mock_wait_us: int, stream: int,
              start_event: int, end_event: int) -> tuple[float, float, float, float]:
    t0 = time.perf_counter()
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
    t1 = time.perf_counter()
    mod.check(ret, "aclnnMockKVSelectGetWorkspaceSize")

    mod.check(acl.rt.record_event(start_event, stream), "acl.rt.record_event(start)")
    t2 = time.perf_counter()
    ret = libs.opapi.aclnnMockKVSelect(ctypes.c_void_p(), workspace_size, executor, ctypes.c_void_p(int(stream)))
    t3 = time.perf_counter()
    mod.check(ret, "aclnnMockKVSelect")
    mod.check(acl.rt.record_event(end_event, stream), "acl.rt.record_event(end)")
    mod.check(acl.rt.synchronize_event(end_event), "acl.rt.synchronize_event(end)")
    elapsed_ms, ret = acl.rt.event_elapsed_time(start_event, end_event)
    mod.check(ret, "acl.rt.event_elapsed_time")
    t4 = time.perf_counter()

    get_workspace_us = (t1 - t0) * 1_000_000.0
    launch_return_us = (t3 - t2) * 1_000_000.0
    event_elapsed_us = float(elapsed_ms) * 1000.0
    total_sync_us = (t4 - t0) * 1_000_000.0
    return get_workspace_us, launch_return_us, event_elapsed_us, total_sync_us


def run_case(args: argparse.Namespace) -> None:
    mod = load_mock_acl_module()
    libs = mod.AclnnLibs(args.opapi)
    mod.check(acl.init(), "acl.init")
    mod.check(acl.rt.set_device(args.device_id), "acl.rt.set_device")
    stream, ret = acl.rt.create_stream()
    mod.check(ret, "acl.rt.create_stream")
    start_event, ret = acl.rt.create_event()
    mod.check(ret, "acl.rt.create_event(start)")
    end_event, ret = acl.rt.create_event()
    mod.check(ret, "acl.rt.create_event(end)")

    inputs = []
    outputs = []
    try:
        input_specs, output_specs = mod.make_specs(
            args.batch_size,
            args.seq_len,
            args.head_num,
            args.head_dim,
            args.max_seq_len,
            args.topk,
            args.block_size,
        )
        inputs = [mod.AclTensor.create(libs, dtype, shape) for dtype, shape in input_specs]
        outputs = [mod.AclTensor.create(libs, dtype, shape) for dtype, shape in output_specs]

        for _ in range(args.warmup):
            call_once(mod, libs, inputs, outputs, args.block_size, args.mock_wait_us,
                      stream, start_event, end_event)

        get_workspace_us = []
        launch_return_us = []
        event_elapsed_us = []
        total_sync_us = []
        for _ in range(args.iters):
            g, l, e, t = call_once(mod, libs, inputs, outputs, args.block_size, args.mock_wait_us,
                                   stream, start_event, end_event)
            get_workspace_us.append(g)
            launch_return_us.append(l)
            event_elapsed_us.append(e)
            total_sync_us.append(t)

        print(
            f"case bs={args.batch_size} seq={args.seq_len} heads={args.head_num} dim={args.head_dim} "
            f"max_seq={args.max_seq_len} topk={args.topk} block_size={args.block_size} "
            f"mock_wait_us={args.mock_wait_us} "
            f"warmup={args.warmup} iters={args.iters}"
        )
        print(f"get_workspace_host_us: {summarize_us(get_workspace_us)}")
        print(f"launch_return_host_us: {summarize_us(launch_return_us)}")
        print(f"event_elapsed_us: {summarize_us(event_elapsed_us)}")
        print(f"total_sync_wall_us: {summarize_us(total_sync_us)}")
    finally:
        for tensor in inputs + outputs:
            tensor.destroy()
        acl.rt.destroy_event(start_event)
        acl.rt.destroy_event(end_event)
        acl.rt.destroy_stream(stream)
        acl.rt.reset_device(args.device_id)
        acl.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure standalone MockKVSelect AICPU latency.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--head-num", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--mock-wait-us", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--opapi", type=Path, default=None)
    args = parser.parse_args()
    if args.opapi is None:
        args.opapi = load_mock_acl_module().default_opapi_path()
    return args


if __name__ == "__main__":
    run_case(parse_args())
