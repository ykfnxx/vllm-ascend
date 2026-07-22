#!/usr/bin/env python3
# coding=utf-8
"""Benchmark npu_gather_selection_kv_cache on NPU (bf16, BSND decode).

Offload path (default on): full cache uses ``empty_with_swapped_memory`` + ``fill_(0)`` +
per-request ``add_(numpy→npu)``, matching ``op/examples/test_npu_gather_selection_kv_cache.py``.

``--topk-reuse-rate`` controls HBM selection-pool hit ratio per timed iteration:
0.0 reinit block_status and all-new topk (read from host/swap);
1.0 keep status and topk (pool hit after the first gather);
(0,1) keep reuse_rate of topk slots and replace the rest.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch_npu

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import custom_ops  # noqa: F401
except ImportError as exc:
    raise SystemExit("custom_ops required; build op/torch_ops_extension first.") from exc

DTYPE = torch.bfloat16
DEFAULT_OUT = Path(__file__).resolve().parent

K_ROPE_DIM = 64
KV_CACHE_DIM = 512
SELECTION_TOPK = 2048
SELECTION_TOPK_BLOCK_SIZE = 1
SELECTION_BLOCK_SIZE = 128
FULL_BLOCK_SIZE = 128
SEQ_LEN = 1
HEAD_NUM = 1

WARMUP = 3
ITERS = 20
BATCH_SIZES = [1, 2, 4, 8]
MAX_SEQ_LENS = [16_384, 32_768, 65_536, 131_072]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def random_selection_topk(all_topk_nums, batch_size, seq_len, head_num, topk):
    out = np.zeros((batch_size, seq_len, head_num, topk), dtype=np.int32)
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(head_num):
                out[b, s, h] = np.random.choice(all_topk_nums, size=topk, replace=False)
    return out


def build_topk_pair(batch_size, max_seq_len, topk, topk_block_size, reuse_rate):
    """Return (prime, bench); reuse_rate keeps that fraction of topk slots on bench."""
    n_blocks = (max_seq_len + topk_block_size - 1) // topk_block_size
    all_ids = np.arange(0, n_blocks, dtype=np.int32)
    prime = random_selection_topk(all_ids, batch_size, SEQ_LEN, HEAD_NUM, topk)
    bench = prime.copy()
    n_replace = int(round(topk * (1.0 - reuse_rate)))
    for b in range(batch_size):
        usable = list(set(all_ids) - set(prime[b].ravel()))
        pick = min(n_replace, len(usable), topk)
        if pick == 0:
            continue
        slots = np.random.choice(topk, size=pick, replace=False)
        bench[b, 0, 0, slots] = np.random.choice(usable, size=pick, replace=False)
    return prime, bench


def build_host_buffers(batch_size, max_seq_len, topk_indices):
    """Same tensor layout as test_gather_selection_kv_cache_eager (host numpy)."""
    s_blocks = (SELECTION_TOPK * SELECTION_TOPK_BLOCK_SIZE + SELECTION_BLOCK_SIZE - 1) // SELECTION_BLOCK_SIZE
    f_blocks = (max_seq_len + FULL_BLOCK_SIZE - 1) // FULL_BLOCK_SIZE
    sel_rows = s_blocks * batch_size * SEQ_LEN * HEAD_NUM

    host = {
        "selection_k_rope": np.random.uniform(size=(sel_rows, SELECTION_BLOCK_SIZE, K_ROPE_DIM)).astype(
            np.float16
        ),
        "selection_kv_cache": np.random.uniform(
            size=(sel_rows, SELECTION_BLOCK_SIZE, KV_CACHE_DIM)
        ).astype(np.float16),
        "selection_kv_block_table": np.arange(0, batch_size * SEQ_LEN * HEAD_NUM * s_blocks, dtype=np.int32).reshape(
            batch_size * SEQ_LEN * HEAD_NUM, s_blocks
        ),
        "selection_kv_block_status": np.full(
            (batch_size, SEQ_LEN, HEAD_NUM, SELECTION_TOPK + 1), -1, dtype=np.int32
        ),
        "selection_topk_indices": topk_indices,
        "full_k_rope": np.random.uniform(size=(f_blocks * batch_size, FULL_BLOCK_SIZE, K_ROPE_DIM)).astype(
            np.float16
        ),
        "full_kv_cache": np.random.uniform(size=(f_blocks * batch_size, FULL_BLOCK_SIZE, KV_CACHE_DIM)).astype(
            np.float16
        ),
        "full_kv_block_table": np.arange(f_blocks * batch_size, dtype=np.int32).reshape(
            batch_size, f_blocks
        ),
        "full_kv_actual_seq": np.full((batch_size,), max_seq_len, dtype=np.int32),
        "full_q_actual_seq": np.full((batch_size,), SEQ_LEN, dtype=np.int32),
    }
    return host


def init_swapped_full_from_host(
    host_arr: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Align with op example ``is_offload``: swap tensor + fill_(0) + add_ one request at a time."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n_rows = host_arr.shape[0]
    rows_per_req, rem = divmod(n_rows, batch_size)
    if rem != 0:
        raise ValueError("full cache row count must be divisible by batch_size")
    out = torch_npu.empty_with_swapped_memory(host_arr.shape, dtype=DTYPE, device=device)
    out.fill_(0)
    for b in range(batch_size):
        start = b * rows_per_req
        end = start + rows_per_req
        out[start:end].add_(torch.from_numpy(host_arr[start:end].copy()).to(DTYPE).to(device=device))
    torch.npu.synchronize()
    return out


def make_inputs(
    batch_size,
    max_seq_len,
    topk_indices,
    device,
    offload_full_cache: bool = True,
):
    host = build_host_buffers(batch_size, max_seq_len, topk_indices)
    inputs = {
        k: torch.from_numpy(host[k].copy()).to(device=device)
        for k in (
            "selection_kv_block_table",
            "selection_kv_block_status",
            "selection_topk_indices",
            "full_kv_block_table",
            "full_kv_actual_seq",
            "full_q_actual_seq",
        )
    }
    # Selection buffers live on HBM; gather writes top-k payload here from swap-backed full cache.
    for k in ("selection_k_rope", "selection_kv_cache"):
        inputs[k] = torch.from_numpy(host[k].copy()).to(DTYPE).to(device=device)
    for k in ("full_k_rope", "full_kv_cache"):
        if offload_full_cache:
            inputs[k] = init_swapped_full_from_host(host[k], batch_size, device)
        else:
            inputs[k] = torch.from_numpy(host[k].copy()).to(DTYPE).to(device=device)
    return inputs


def reinit_selection_kv_block_status(inputs: dict) -> None:
    """Clear cross-step reuse metadata (same as OffloadCache.reinit_status): all slots empty."""
    inputs["selection_kv_block_status"].fill_(-1)


def advance_topk_indices(
    inputs: dict,
    batch_size: int,
    max_seq_len: int,
    reuse_rate: float,
    *,
    topk: int = SELECTION_TOPK,
    topk_block_size: int = SELECTION_TOPK_BLOCK_SIZE,
) -> None:
    """Update topk: keep reuse_rate fraction of slots, draw new global ids for the rest."""
    reuse_rate = max(0.0, min(1.0, reuse_rate))
    n_replace = int(round(topk * (1.0 - reuse_rate)))
    if n_replace <= 0:
        return
    n_blocks = (max_seq_len + topk_block_size - 1) // topk_block_size
    all_ids = np.arange(0, n_blocks, dtype=np.int32)
    if n_replace >= topk:
        topk_np = random_selection_topk(all_ids, batch_size, SEQ_LEN, HEAD_NUM, topk)
        inputs["selection_topk_indices"].copy_(
            torch.from_numpy(topk_np).to(device=inputs["selection_topk_indices"].device)
        )
        return
    cur = inputs["selection_topk_indices"].detach().cpu().numpy()
    out = cur.copy()
    for b in range(batch_size):
        for s in range(SEQ_LEN):
            for h in range(HEAD_NUM):
                slots = np.random.choice(topk, size=n_replace, replace=False) # 随机选 n_replace 个槽位要换
                kept = np.delete(out[b, s, h], slots) # 其余槽位上的 id 保留
                usable = list(set(all_ids) - set(kept)) # 新 id 不能和「保留的 id」重复
                pick = min(n_replace, len(usable))
                if pick > 0:
                    out[b, s, h, slots[:pick]] = np.random.choice(usable, size=pick, replace=False) # 从 usable 中随机选 pick 个 id 替换 slots 中的 id
    inputs["selection_topk_indices"].copy_(
        torch.from_numpy(out).to(device=inputs["selection_topk_indices"].device)
    )


def prepare_gather_step(
    inputs: dict,
    batch_size: int,
    max_seq_len: int,
    reuse_rate: float,
) -> None:
    """Per-iteration setup: reuse_rate=0 clears pool metadata; else roll topk by miss fraction."""
    if reuse_rate <= 0.0:
        reinit_selection_kv_block_status(inputs)
    advance_topk_indices(inputs, batch_size, max_seq_len, reuse_rate)


def run_gather(inputs):
    return torch_npu.npu_gather_selection_kv_cache(
        selection_k_rope=inputs["selection_k_rope"],
        selection_kv_cache=inputs["selection_kv_cache"],
        selection_kv_block_table=inputs["selection_kv_block_table"],
        selection_kv_block_status=inputs["selection_kv_block_status"],
        selection_topk_indices=inputs["selection_topk_indices"],
        full_k_rope=inputs["full_k_rope"],
        full_kv_cache=inputs["full_kv_cache"],
        full_kv_block_table=inputs["full_kv_block_table"],
        full_kv_actual_seq=inputs["full_kv_actual_seq"],
        full_q_actual_seq=inputs["full_q_actual_seq"],
        selection_topk_block_size=SELECTION_TOPK_BLOCK_SIZE,
    )


def measure_ms(fn, warmup, iters, *, before_each=None):
    """Time `fn`. If `before_each` is set, call it before every warmup/timed iteration."""
    for _ in range(warmup):
        if before_each is not None:
            before_each()
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        if before_each is not None:
            before_each()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return np.asarray(samples, dtype=np.float64)


def benchmark(
    batch_size,
    max_seq_len,
    device,
    warmup,
    iters,
    offload,
    reuse_rate,
):
    n_blocks = (max_seq_len + SELECTION_TOPK_BLOCK_SIZE - 1) // SELECTION_TOPK_BLOCK_SIZE
    init_topk = random_selection_topk(
        np.arange(0, n_blocks, dtype=np.int32), batch_size, SEQ_LEN, HEAD_NUM, SELECTION_TOPK
    )
    inputs = make_inputs(batch_size, max_seq_len, init_topk, device, offload)

    def _before_each():
        prepare_gather_step(inputs, batch_size, max_seq_len, reuse_rate)

    lat = measure_ms(lambda: run_gather(inputs), warmup, iters, before_each=_before_each)
    payload = (
        batch_size
        * SEQ_LEN
        * HEAD_NUM
        * SELECTION_TOPK
        * SELECTION_TOPK_BLOCK_SIZE
        * (K_ROPE_DIM + KV_CACHE_DIM)
        * 2
    )
    host_frac = max(0.0, min(1.0, 1.0 - reuse_rate))
    avg_ms = float(lat.mean())
    gbps = (payload * host_frac / (1024**3)) / (avg_ms / 1000.0)
    return {
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "offload": offload,
        "reuse_rate": reuse_rate,
        "avg_ms": avg_ms,
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p99_ms": float(np.percentile(lat, 99)),
        "gbps_est": gbps,
    }


def parse_args():
    p = argparse.ArgumentParser(description="gather_selection_kv_cache perf (bf16)")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--iters", type=int, default=ITERS)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=16_384)
    p.add_argument(
        "--topk-reuse-rate",
        type=float,
        default=0.0,
        help="HBM selection-pool hit ratio target per iter (0=all from host/swap, 1=all pool hits)",
    )
    p.add_argument(
        "--offload-full-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store full_kv on host swap via empty_with_swapped_memory (default: on)",
    )
    p.add_argument("--sweep", action="store_true")
    p.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes for --sweep (default: 1,2,4,8)",
    )
    p.add_argument(
        "--max-seq-lens",
        type=str,
        default=None,
        help="Comma-separated max_seq_len for --sweep (default: 16384,32768,65536,131072)",
    )
    p.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Sweep CSV/PNG basename without extension (default: auto from offload/reuse/seq tags)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.npu.is_available():
        raise SystemExit("NPU not available")

    device = torch.device(args.device)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.set_device(device)

    kw = dict(
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        offload=args.offload_full_cache,
        reuse_rate=args.topk_reuse_rate,
    )

    if args.sweep:
        batch_sizes = _parse_int_list(args.batch_sizes) if args.batch_sizes else BATCH_SIZES
        max_seq_lens = _parse_int_list(args.max_seq_lens) if args.max_seq_lens else MAX_SEQ_LENS
        rows = []
        for max_seq in max_seq_lens:
            for bs in batch_sizes:
                print(f"bs={bs} max_seq={max_seq} ...", flush=True)
                try:
                    row = benchmark(bs, max_seq, **kw)
                    row["error"] = ""
                    print(f"  avg={row['avg_ms']:.3f}ms", flush=True)
                except RuntimeError as exc:
                    row = {
                        "batch_size": bs,
                        "max_seq_len": max_seq,
                        "offload": args.offload_full_cache,
                        "reuse_rate": args.topk_reuse_rate,
                        "avg_ms": float("nan"),
                        "p50_ms": float("nan"),
                        "p90_ms": float("nan"),
                        "p99_ms": float("nan"),
                        "gbps_est": float("nan"),
                        "error": str(exc)[:300],
                    }
                    print(f"  FAIL: {row['error'][:120]}", flush=True)
                rows.append(row)
        out = args.out_dir
        tag = "offload_on" if args.offload_full_cache else "offload_off"
        reuse_tag = f"reuse{args.topk_reuse_rate:g}".replace(".", "p")
        seq_tag = "_".join(str(s) for s in max_seq_lens)
        stem = args.output_stem or f"gather_selection_kv_cache_latency_sweep_{tag}_{reuse_tag}_seq{seq_tag}"
        csv_path = out / f"{stem}.csv"
        png_path = out / f"{stem}.png"
        fieldnames = [
            "batch_size",
            "max_seq_len",
            "offload",
            "reuse_rate",
            "avg_ms",
            "p50_ms",
            "p90_ms",
            "p99_ms",
            "gbps_est",
            "error",
        ]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        by_seq = defaultdict(list)
        for r in rows:
            if r.get("error"):
                continue
            by_seq[r["max_seq_len"]].append(r)
        if plt is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            for seq in sorted(by_seq):
                pts = sorted(by_seq[seq], key=lambda r: r["batch_size"])
                ax.plot(
                    [p["batch_size"] for p in pts],
                    [p["avg_ms"] for p in pts],
                    marker="o",
                    label=f"seq={seq}",
                )
            ax.set_xlabel("batch_size")
            ax.set_ylabel("avg latency (ms)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels([str(b) for b in batch_sizes])
            ax.set_title(f"gather_selection_kv_cache ({tag}, reuse={args.topk_reuse_rate:g}, bf16)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
        else:
            png_path = None
        print(f"csv: {csv_path}")
        if png_path is not None:
            print(f"png: {png_path}")
        return

    row = benchmark(args.batch_size, args.max_seq_len, **kw)
    print(
        f"bs={row['batch_size']} max_seq={row['max_seq_len']} offload={row['offload']} "
        f"reuse={row['reuse_rate']:.2f} bf16: "
        f"avg={row['avg_ms']:.3f}ms p99={row['p99_ms']:.3f}ms gbps~{row['gbps_est']:.1f}"
    )


if __name__ == "__main__":
    main()
