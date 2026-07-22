#!/usr/bin/env python3
# coding=utf-8
"""Benchmark segmented GatherSelectionKvCache: KVSelect + KVGather."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch_npu

from test_npu_gather_selection_kv_cache_perf import (
    HEAD_NUM,
    SELECTION_TOPK,
    SELECTION_TOPK_BLOCK_SIZE,
    SEQ_LEN,
    advance_topk_indices,
    make_inputs,
    random_selection_topk,
    reinit_selection_kv_block_status,
)

try:
    import custom_ops  # noqa: F401
except ImportError as exc:
    raise SystemExit("custom_ops required; build op/torch_ops_extension first.") from exc


DEFAULT_OUT = Path(__file__).resolve().parent


def make_workspace(inputs: dict, batch_size: int, device: torch.device) -> dict:
    topk_shape = inputs["selection_topk_indices"].shape
    row_shape = (batch_size * SEQ_LEN * HEAD_NUM,)
    return {
        "hit_sparse_indices": torch.empty(topk_shape, dtype=torch.int32, device=device),
        "miss_topk_indices": torch.empty(topk_shape, dtype=torch.int32, device=device),
        "miss_insert_indices": torch.empty(topk_shape, dtype=torch.int32, device=device),
        "hit_actual_seq": torch.empty(row_shape, dtype=torch.int32, device=device),
        "miss_actual_seq": torch.empty(row_shape, dtype=torch.int32, device=device),
        "miss_count": torch.empty(row_shape, dtype=torch.int32, device=device),
        "hit_count": torch.empty(row_shape, dtype=torch.int32, device=device),
        "selection_status_empty": torch.empty(row_shape, dtype=torch.int32, device=device),
        "selection_kv_actual_seq": torch.empty(row_shape, dtype=torch.int32, device=device),
    }


def run_kv_select(inputs: dict, ws: dict) -> None:
    torch_npu.npu_kv_select_out(
        inputs["selection_k_rope"],
        inputs["selection_kv_cache"],
        inputs["selection_kv_block_table"],
        inputs["selection_kv_block_status"],
        inputs["selection_topk_indices"],
        inputs["full_k_rope"],
        inputs["full_kv_cache"],
        inputs["full_kv_block_table"],
        inputs["full_kv_actual_seq"],
        inputs["full_q_actual_seq"],
        ws["hit_sparse_indices"],
        ws["miss_topk_indices"],
        ws["miss_insert_indices"],
        ws["hit_actual_seq"],
        ws["miss_actual_seq"],
        ws["miss_count"],
        ws["hit_count"],
        ws["selection_status_empty"],
        selection_topk_block_size=SELECTION_TOPK_BLOCK_SIZE,
    )


def run_kv_gather(inputs: dict, ws: dict) -> None:
    torch_npu.npu_kv_gather_out(
        inputs["selection_k_rope"],
        inputs["selection_kv_cache"],
        inputs["selection_kv_block_table"],
        inputs["selection_kv_block_status"],
        ws["miss_topk_indices"],
        ws["miss_insert_indices"],
        inputs["full_k_rope"],
        inputs["full_kv_cache"],
        inputs["full_kv_block_table"],
        inputs["full_kv_actual_seq"],
        inputs["full_q_actual_seq"],
        ws["hit_actual_seq"],
        ws["miss_actual_seq"],
        ws["miss_count"],
        ws["hit_count"],
        ws["selection_status_empty"],
        ws["selection_kv_actual_seq"],
        selection_topk_block_size=SELECTION_TOPK_BLOCK_SIZE,
    )


def prepare_step(inputs: dict, batch_size: int, max_seq_len: int, reuse_rate: float) -> None:
    if reuse_rate <= 0.0:
        reinit_selection_kv_block_status(inputs)
    advance_topk_indices(inputs, batch_size, max_seq_len, reuse_rate)


def time_one_step(inputs: dict, ws: dict) -> tuple[float, float, float]:
    start_select = torch.npu.Event(enable_timing=True)
    end_select = torch.npu.Event(enable_timing=True)
    end_gather = torch.npu.Event(enable_timing=True)
    start_select.record()
    run_kv_select(inputs, ws)
    end_select.record()
    run_kv_gather(inputs, ws)
    end_gather.record()
    end_gather.synchronize()
    select_ms = start_select.elapsed_time(end_select)
    gather_ms = end_select.elapsed_time(end_gather)
    return select_ms, gather_ms, select_ms + gather_ms


def benchmark(
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    warmup: int,
    iters: int,
    offload: bool,
    reuse_rate: float,
) -> dict:
    n_blocks = (max_seq_len + SELECTION_TOPK_BLOCK_SIZE - 1) // SELECTION_TOPK_BLOCK_SIZE
    init_topk = random_selection_topk(
        np.arange(0, n_blocks, dtype=np.int32), batch_size, SEQ_LEN, HEAD_NUM, SELECTION_TOPK
    )
    inputs = make_inputs(batch_size, max_seq_len, init_topk, device, offload)
    ws = make_workspace(inputs, batch_size, device)

    for _ in range(warmup):
        prepare_step(inputs, batch_size, max_seq_len, reuse_rate)
        time_one_step(inputs, ws)
    torch.npu.synchronize()

    samples = np.empty((iters, 3), dtype=np.float64)
    for i in range(iters):
        prepare_step(inputs, batch_size, max_seq_len, reuse_rate)
        samples[i] = time_one_step(inputs, ws)

    status = inputs["selection_kv_block_status"][..., :SELECTION_TOPK].reshape(-1, SELECTION_TOPK)
    filled_slots = float((status >= 0).float().sum(dim=1).mean().item())
    actual_seq = float(ws["selection_kv_actual_seq"].float().mean().item())
    return {
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "offload": offload,
        "reuse_rate": reuse_rate,
        "select_avg_ms": float(samples[:, 0].mean()),
        "gather_avg_ms": float(samples[:, 1].mean()),
        "total_avg_ms": float(samples[:, 2].mean()),
        "select_p99_ms": float(np.percentile(samples[:, 0], 99)),
        "gather_p99_ms": float(np.percentile(samples[:, 1], 99)),
        "total_p99_ms": float(np.percentile(samples[:, 2], 99)),
        "filled_slots": filled_slots,
        "actual_seq": actual_seq,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KVSelect + KVGather perf (bf16)")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=131072)
    p.add_argument("--topk-reuse-rate", type=float, default=0.0)
    p.add_argument("--offload-full-cache", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.npu.is_available():
        raise SystemExit("NPU not available")

    device = torch.device(args.device)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.set_device(device)

    row = benchmark(
        args.batch_size,
        args.max_seq_len,
        device,
        args.warmup,
        args.iters,
        args.offload_full_cache,
        args.topk_reuse_rate,
    )
    print(
        f"bs={row['batch_size']} max_seq={row['max_seq_len']} offload={row['offload']} "
        f"reuse={row['reuse_rate']:.4f} "
        f"select={row['select_avg_ms']:.4f}ms gather={row['gather_avg_ms']:.4f}ms "
        f"total={row['total_avg_ms']:.4f}ms "
        f"p99=({row['select_p99_ms']:.4f},{row['gather_p99_ms']:.4f},{row['total_p99_ms']:.4f})ms "
        f"filled_slots={row['filled_slots']:.1f} actual_seq={row['actual_seq']:.1f}"
    )


if __name__ == "__main__":
    main()
