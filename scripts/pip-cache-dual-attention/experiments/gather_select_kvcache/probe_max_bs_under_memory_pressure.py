#!/usr/bin/env python3
# coding=utf-8
"""Fill NPU memory to target utilization, then find max batch_size for gather at long context."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_npu_gather_selection_kv_cache_perf import (  # noqa: E402
    SELECTION_TOPK,
    SELECTION_TOPK_BLOCK_SIZE,
    build_topk_pair,
    make_inputs,
    run_gather,
)

try:
    import custom_ops  # noqa: F401
except ImportError as exc:
    raise SystemExit("custom_ops required") from exc


def npu_mem_stats(device: torch.device) -> tuple[int, int, int]:
    torch.npu.synchronize()
    allocated = torch.npu.memory_allocated(device)
    reserved = torch.npu.memory_reserved(device)
    total = torch.npu.get_device_properties(device).total_memory
    return allocated, reserved, total


def fill_memory_to_target(device: torch.device, target_ratio: float) -> list[torch.Tensor]:
    holders: list[torch.Tensor] = []
    allocated, _, total = npu_mem_stats(device)
    goal = int(total * target_ratio)
    chunk_elems = 256 * 1024 * 1024 // 2

    print(f"NPU total={total / 1024**3:.2f} GiB goal={goal / 1024**3:.2f} GiB ({target_ratio * 100:.0f}%)")
    while allocated < goal:
        try:
            t = torch.empty(chunk_elems, dtype=torch.bfloat16, device=device)
            holders.append(t)
            allocated, reserved, _ = npu_mem_stats(device)
            if len(holders) % 8 == 0:
                print(
                    f"  filler chunks={len(holders)} allocated={allocated / 1024**3:.2f} GiB "
                    f"reserved={reserved / 1024**3:.2f} GiB",
                    flush=True,
                )
        except RuntimeError as e:
            print(f"  filler stopped: {e}", flush=True)
            break

    allocated, reserved, total = npu_mem_stats(device)
    print(
        f"Filler done: allocated={allocated / 1024**3:.2f} GiB "
        f"({100 * allocated / total:.1f}%) reserved={reserved / 1024**3:.2f} GiB"
    )
    return holders


def try_batch(
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    *,
    offload: bool,
) -> tuple[bool, str]:
    _, bench = build_topk_pair(batch_size, max_seq_len, SELECTION_TOPK, SELECTION_TOPK_BLOCK_SIZE, 0.0)
    try:
        gc.collect()
        torch.npu.empty_cache()
        inputs = make_inputs(batch_size, max_seq_len, bench, device, offload)
        torch.npu.synchronize()
        run_gather(inputs)
        torch.npu.synchronize()
        del inputs
        gc.collect()
        torch.npu.empty_cache()
        return True, "ok"
    except RuntimeError as e:
        gc.collect()
        torch.npu.empty_cache()
        return False, str(e)[:200]


def find_max_batch(
    max_seq_len: int,
    device: torch.device,
    *,
    offload: bool,
    max_try: int,
    min_bs: int = 1,
) -> tuple[int, dict[int, str]]:
    errors: dict[int, str] = {}
    min_bs = max(1, min_bs)
    ok, msg = try_batch(min_bs, max_seq_len, device, offload=offload)
    errors[min_bs] = msg
    print(f"  try bs={min_bs}: {'OK' if ok else 'FAIL'} {msg[:80] if not ok else ''}", flush=True)
    if not ok:
        return min_bs - 1 if min_bs > 1 else 0, errors

    lo = min_bs
    hi = min_bs * 2

    while hi <= max_try:
        ok, msg = try_batch(hi, max_seq_len, device, offload=offload)
        errors[hi] = msg
        print(f"  try bs={hi}: {'OK' if ok else 'FAIL'} {msg[:80] if not ok else ''}", flush=True)
        if ok:
            lo = hi
            hi *= 2
        else:
            break

    if hi > max_try:
        return lo, errors

    left, right = lo, hi
    while left + 1 < right:
        mid = (left + right) // 2
        ok, msg = try_batch(mid, max_seq_len, device, offload=offload)
        errors[mid] = msg
        print(f"  try bs={mid}: {'OK' if ok else 'FAIL'}", flush=True)
        if ok:
            left = mid
        else:
            right = mid
    return left, errors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="npu:0")
    p.add_argument("--max-seq-len", type=int, default=1_000_000)
    p.add_argument("--mem-target", type=float, default=0.90)
    p.add_argument("--max-try-bs", type=int, default=512)
    p.add_argument("--min-bs", type=int, default=1, help="Start exponential search at this batch size")
    p.add_argument("--skip-fill", action="store_true")
    p.add_argument(
        "--offload-only",
        choices=("off", "on", "both"),
        default="both",
        help="Probe only offload=OFF, only ON, or both (default both)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.npu.is_available():
        raise SystemExit("NPU not available")

    device = torch.device(args.device)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.set_device(device)

    alloc0, _, total = npu_mem_stats(device)
    print(f"Baseline allocated={alloc0 / 1024**3:.3f} GiB / {total / 1024**3:.2f} GiB")

    fillers: list[torch.Tensor] = []
    if not args.skip_fill:
        fillers = fill_memory_to_target(device, args.mem_target)

    modes = {"off": [False], "on": [True], "both": [False, True]}[args.offload_only]
    results = {}
    for offload in modes:
        gc.collect()
        torch.npu.empty_cache()
        alloc, reserved, _ = npu_mem_stats(device)
        print(
            f"\n--- before probe: allocated={alloc / 1024**3:.3f} GiB "
            f"reserved={reserved / 1024**3:.3f} GiB ---",
            flush=True,
        )
        label = "offload=ON" if offload else "offload=OFF"
        print(f"\n======== {label} max_seq_len={args.max_seq_len} ========", flush=True)
        max_bs, errs = find_max_batch(
            args.max_seq_len,
            device,
            offload=offload,
            max_try=args.max_try_bs,
            min_bs=args.min_bs,
        )
        results[label] = max_bs
        print(f"{label}: max_batch_size={max_bs}", flush=True)
        fail_bs = max_bs + 1
        if fail_bs in errs:
            print(f"  bs={fail_bs} (fail): {errs[fail_bs]}")

    print("\n======== SUMMARY ========")
    print(
        f"mem_target={args.mem_target * 100:.0f}% max_seq_len={args.max_seq_len} "
        f"min_bs={args.min_bs} bf16"
    )
    for k, v in results.items():
        print(f"  {k}: max_bs={v}")

    del fillers
    gc.collect()


if __name__ == "__main__":
    main()
