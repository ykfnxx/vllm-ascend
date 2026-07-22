#!/usr/bin/env python3
# coding=utf-8
"""Benchmark torch_npu.npu_sparse_flash_attention latency on NPU.

Shapes and hyperparameters follow ``op/examples/test_npu_sparse_flash_attention.py``
``test_sfa_eager`` (BSND query/KV, sparse_mode=3, rope). ``sparse_block_count`` is
fixed to 2048 (top-k blocks); sweep ``batch_size``.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_npu

DEFAULT_OUT = Path(__file__).resolve().parent

# Align with test_sfa_eager in op/examples/test_npu_sparse_flash_attention.py
SCALE_VALUE = 0.041666666666666664  # 1 / sqrt(512 + 64)
SPARSE_BLOCK_SIZE = 1
SPARSE_BLOCK_COUNT = 2048  # top-k
S1 = 1
N1 = 128
N2 = 1
DN = 512
DR = 64
QUERY_DTYPE = torch.float16

BATCH_SIZES_DEFAULT = [1, 2, 4, 8]
KEY_SEQ_LENS_DEFAULT = [16_384, 32_768, 65_536, 131_072]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]
WARMUP = 3
ITERS = 20
RNG_SEED = 21


class SfaInputs(NamedTuple):
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    sparse_indices: torch.Tensor
    query_rope: torch.Tensor
    key_rope: torch.Tensor
    act_seq_q: torch.Tensor
    act_seq_kv: torch.Tensor


def run_npu_sparse_flash_attention(query, key, value, sparse_indices, scale_value, *, sparse_block_size=1, **kwargs):
    """torch_npu built-in SFA; sparse_block_size is keyword-only; attention out is [0] if tuple."""
    if kwargs.get("query_rope") is not None or kwargs.get("key_rope") is not None:
        kwargs.setdefault("attention_mode", 2)
    out = torch_npu.npu_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        sparse_block_size=sparse_block_size,
        **kwargs,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


def build_sparse_indices(batch_size: int, s1: int, n2: int, topk: int, s2_act: int) -> torch.Tensor:
    """Per-(batch, seq, kv_head) block ids in [0, s2_act - s1] inclusive, without replacement."""
    upper = s2_act - s1 + 1
    if topk > upper:
        raise ValueError(f"topk={topk} exceeds available sparse id range size {upper}")
    arr = np.zeros((batch_size, s1, n2, topk), dtype=np.int32)
    for b in range(batch_size):
        for si in range(s1):
            for hi in range(n2):
                arr[b, si, hi] = random.sample(range(upper), topk)
    return torch.from_numpy(arr)


def make_inputs(batch_size: int, key_seq_len: int, device: torch.device) -> SfaInputs:
    """BSND SFA: key tensor length ``key_seq_len``, actual KV len = ``key_seq_len``."""
    rng = np.random.default_rng(RNG_SEED + batch_size + key_seq_len)
    query = torch.tensor(rng.uniform(-10, 10, (batch_size, S1, N1, DN)), dtype=QUERY_DTYPE, device=device)
    key = torch.tensor(rng.uniform(-5, 10, (batch_size, key_seq_len, N2, DN)), dtype=QUERY_DTYPE, device=device)
    value = key.clone()
    sparse_indices = build_sparse_indices(batch_size, S1, N2, SPARSE_BLOCK_COUNT, key_seq_len).to(
        device=device
    )
    query_rope = torch.tensor(rng.uniform(-10, 10, (batch_size, S1, N1, DR)), dtype=QUERY_DTYPE, device=device)
    key_rope = torch.tensor(
        rng.uniform(-10, 10, (batch_size, key_seq_len, N2, DR)), dtype=QUERY_DTYPE, device=device
    )
    act_seq_q = torch.full((batch_size,), S1, dtype=torch.int32, device=device)
    act_seq_kv = torch.full((batch_size,), key_seq_len, dtype=torch.int32, device=device)
    return SfaInputs(query, key, value, sparse_indices, query_rope, key_rope, act_seq_q, act_seq_kv)


def forward_sfa(inp: SfaInputs) -> torch.Tensor:
    return run_npu_sparse_flash_attention(
        inp.query,
        inp.key,
        inp.value,
        inp.sparse_indices,
        SCALE_VALUE,
        sparse_block_size=SPARSE_BLOCK_SIZE,
        actual_seq_lengths_query=inp.act_seq_q,
        actual_seq_lengths_kv=inp.act_seq_kv,
        query_rope=inp.query_rope,
        key_rope=inp.key_rope,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        block_table=None,
    )


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


def benchmark_batch(
    batch_size: int, key_seq_len: int, device: torch.device, warmup: int, iters: int
) -> dict:
    inputs = make_inputs(batch_size, key_seq_len, device)
    lat = measure_ms(lambda: forward_sfa(inputs), warmup, iters)
    return {
        "batch_size": batch_size,
        "key_cache_len": key_seq_len,
        "topk": SPARSE_BLOCK_COUNT,
        "sparse_block_size": SPARSE_BLOCK_SIZE,
        "s1": S1,
        "n_query_heads": N1,
        "n_kv_heads": N2,
        "head_dim": DN,
        "rope_dim": DR,
        "dtype": str(QUERY_DTYPE).split(".")[-1],
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
    ax.set_title(f"npu_sparse_flash_attention (topk={SPARSE_BLOCK_COUNT}, fp16)")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Average latency (ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Sparse Flash Attention NPU latency vs batch size (topk=2048)")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(x) for x in BATCH_SIZES_DEFAULT),
        help="Comma-separated batch sizes, e.g. 1,2,4,8",
    )
    p.add_argument(
        "--key-seq-lens",
        type=str,
        default=",".join(str(x) for x in KEY_SEQ_LENS_DEFAULT),
        help="Comma-separated key cache lengths (KV sequence dim and actual_seq_lengths_kv)",
    )
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--iters", type=int, default=ITERS)
    p.add_argument("--seed", type=int, default=RNG_SEED, help="Base seed for numpy RNG and Python random")
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib PNG")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.npu.is_available():
        raise SystemExit("NPU not available; source Ascend CANN / torch_npu env first.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    batch_sizes = _parse_int_list(args.batch_sizes)
    key_seq_lens = _parse_int_list(args.key_seq_lens)
    if not batch_sizes:
        raise SystemExit("No batch sizes parsed from --batch-sizes")
    if not key_seq_lens:
        raise SystemExit("No key seq lengths parsed from --key-seq-lens")

    device = torch.device(args.device)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.set_device(device)

    out_dir = args.out_dir
    seq_tag = "_".join(str(s) for s in key_seq_lens)
    stem = f"sparse_flash_attention_latency_sweep_seq{seq_tag}"
    csv_path = out_dir / f"{stem}.csv"
    plot_path = out_dir / f"{stem}.png"

    fieldnames = [
        "batch_size",
        "key_cache_len",
        "topk",
        "sparse_block_size",
        "s1",
        "n_query_heads",
        "n_kv_heads",
        "head_dim",
        "rope_dim",
        "dtype",
        "avg_ms",
        "p50_ms",
        "p90_ms",
        "p99_ms",
        "error",
    ]

    rows: list[dict] = []
    for key_seq_len in key_seq_lens:
        for b in batch_sizes:
            print(f"batch_size={b}, key_seq_len={key_seq_len} (topk={SPARSE_BLOCK_COUNT}) ...", flush=True)
            try:
                row = benchmark_batch(b, key_seq_len, device, args.warmup, args.iters)
                row["error"] = ""
                print(
                    f"  avg={row['avg_ms']:.3f}ms p50={row['p50_ms']:.3f}ms "
                    f"p90={row['p90_ms']:.3f}ms p99={row['p99_ms']:.3f}ms",
                    flush=True,
                )
            except RuntimeError as exc:
                row = {
                    "batch_size": b,
                    "key_cache_len": key_seq_len,
                    "topk": SPARSE_BLOCK_COUNT,
                    "sparse_block_size": SPARSE_BLOCK_SIZE,
                    "s1": S1,
                    "n_query_heads": N1,
                    "n_kv_heads": N2,
                    "head_dim": DN,
                    "rope_dim": DR,
                    "dtype": str(QUERY_DTYPE).split(".")[-1],
                    "avg_ms": float("nan"),
                    "p50_ms": float("nan"),
                    "p90_ms": float("nan"),
                    "p99_ms": float("nan"),
                    "error": str(exc)[:300],
                }
                print(f"  FAIL: {row['error'][:120]}", flush=True)
            rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"csv: {csv_path}")
    if not args.no_plot:
        plot_latency(rows, plot_path, batch_sizes)
        print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
