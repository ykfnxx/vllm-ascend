#!/usr/bin/env python3
# coding=utf-8
"""Sweep Lightning Indexer latency on NPU; write CSV + matplotlib plot."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_npu

DEFAULT_OUT = Path(__file__).resolve().parent
BATCH_SIZES = [1, 2, 4, 8]
SEQ_LENS = [16_384, 32_768, 65_536, 131_072]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]
WARMUP = 3
ITERS = 20
BLOCK_SIZE = 128
SPARSE_COUNT = 2048
SPARSE_MODE = 3
NUM_HEADS = 64
HEAD_DIM = 128
DTYPE = torch.bfloat16


class IndexerInputs(NamedTuple):
    """Tensors for npu_lightning_indexer (TND query, PA_BSND paged key)."""

    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    q_lens: torch.Tensor
    k_lens: torch.Tensor
    block_table: torch.Tensor


def parse_args():
    p = argparse.ArgumentParser(description="Lightning Indexer NPU latency sweep")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--iters", type=int, default=ITERS)
    p.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes (default: 1,2,4,8)",
    )
    p.add_argument(
        "--key-seq-lens",
        type=str,
        default=None,
        help="Comma-separated key cache lengths (default: 16384,32768,65536,131072)",
    )
    return p.parse_args()


def run_indexer(query, key, weights, q_lens, k_lens, block_table):
    out = torch_npu.npu_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=q_lens,
        actual_seq_lengths_key=k_lens,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=SPARSE_MODE,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


def make_inputs(
    batch_size: int,
    key_seq_len: int,
    device: torch.device,
    query_seq_len: int = 1,
) -> IndexerInputs:
    """Decode-shaped TND inputs: B batches × query_seq_len tokens, paged key cache of length key_seq_len."""
    blocks_per_seq = (key_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_key_blocks = batch_size * blocks_per_seq
    token_count = batch_size * query_seq_len

    query = torch.randn(token_count, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    key = torch.randn(num_key_blocks, BLOCK_SIZE, 1, HEAD_DIM, dtype=DTYPE, device=device)
    weights = torch.randn(token_count, NUM_HEADS, dtype=DTYPE, device=device)

    q_lens = torch.cumsum(
        torch.full((batch_size,), query_seq_len, dtype=torch.int32, device=device),
        dim=0,
    ).to(torch.int32)
    k_lens = torch.full((batch_size,), key_seq_len, dtype=torch.int32, device=device)
    block_table = torch.arange(num_key_blocks, dtype=torch.int32, device=device).view(
        batch_size, blocks_per_seq
    )

    return IndexerInputs(query, key, weights, q_lens, k_lens, block_table)

def measure_ms(fn, warmup: int, iters: int) -> np.ndarray:
    for _ in range(warmup):
        fn()

    torch.npu.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return np.asarray(samples, dtype=np.float64)


def benchmark(batch_size: int, seq_len: int, device: torch.device, warmup: int, iters: int) -> dict:
    inputs = make_inputs(batch_size, seq_len, device)
    lat = measure_ms(lambda: run_indexer(*inputs), warmup, iters)
    return {
        "batch_size": batch_size,
        "key_cache_len": seq_len,
        "avg_ms": float(lat.mean()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p99_ms": float(np.percentile(lat, 99)),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_latency(rows: list[dict], path: Path, batch_sizes: list[int]) -> None:
    by_seq: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        by_seq[row["key_cache_len"]].append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    for seq_len in sorted(by_seq):
        pts = sorted(by_seq[seq_len], key=lambda r: r["batch_size"])
        label = f"key_cache_len={seq_len // 1000}k" if seq_len % 1000 == 0 else f"key_cache_len={seq_len}"
        ax.plot(
            [p["batch_size"] for p in pts],
            [p["avg_ms"] for p in pts],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_title("Lightning Indexer Latency vs Batch Size (TND, bf16)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Average Latency (ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("NPU not available; source Ascend env first")

    device = torch.device(args.device)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.set_device(device)
    out_dir = args.out_dir
    batch_sizes = _parse_int_list(args.batch_sizes) if args.batch_sizes else BATCH_SIZES
    seq_lens = _parse_int_list(args.key_seq_lens) if args.key_seq_lens else SEQ_LENS
    seq_tag = "_".join(str(s) for s in seq_lens)
    stem = f"lightning_indexer_latency_sweep_seq{seq_tag}"
    csv_path = out_dir / f"{stem}.csv"
    plot_path = out_dir / f"{stem}.png"

    rows = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            print(f"batch_size={batch_size}, key_seq_len={seq_len} ...", flush=True)
            try:
                row = benchmark(batch_size, seq_len, device, args.warmup, args.iters)
                row["error"] = ""
                print(
                    f"  avg={row['avg_ms']:.3f}ms p50={row['p50_ms']:.3f}ms p99={row['p99_ms']:.3f}ms",
                    flush=True,
                )
            except RuntimeError as exc:
                row = {
                    "batch_size": batch_size,
                    "key_cache_len": seq_len,
                    "avg_ms": float("nan"),
                    "p50_ms": float("nan"),
                    "p90_ms": float("nan"),
                    "p99_ms": float("nan"),
                    "error": str(exc)[:300],
                }
                print(f"  FAIL: {row['error'][:120]}", flush=True)
            rows.append(row)

    fieldnames = ["batch_size", "key_cache_len", "avg_ms", "p50_ms", "p90_ms", "p99_ms", "error"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    plot_latency(rows, plot_path, batch_sizes)
    print(f"csv:  {csv_path}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
