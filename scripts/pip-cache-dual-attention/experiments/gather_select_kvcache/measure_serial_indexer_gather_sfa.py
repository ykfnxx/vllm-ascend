#!/usr/bin/env python3
# coding=utf-8
"""Measure serial LightningIndexer + GatherSelectionKvCache + SparseFlashAttention.

Two cases are measured in order:
  1. bs=64 on the default stream, full cores;
  2. bs=32 on one stream limited to half cores.

The measured serial time is the sum of the three NPU operator event times. Host
topk preparation used to emulate reuse is outside the serial operator sum.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu

THIS_DIR = Path(__file__).resolve().parent
REPO_SRC = THIS_DIR.parent.parent / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from baseline import (  # noqa: E402
    BaselineConfig,
    BaselineRuntime,
    GatherInputs,
    SparseAttnInputs,
    StepMetrics,
)


DEFAULT_OUT = THIS_DIR
DEFAULT_MAX_SEQ_LEN = 16_384
DEFAULT_REUSE_RATE = 0.9
DEFAULT_WARMUP = 2
DEFAULT_ITERS = 5
DEFAULT_STREAM_CUBE_CORES = 10
DEFAULT_STREAM_VECTOR_CORES = 20


class SerialMeasureRuntime(BaselineRuntime):
    """BaselineRuntime variant compatible with the local custom SFA schema."""

    def run_sparse_attn(
        self,
        gather_inputs: GatherInputs,
        sparse_attn_inputs: SparseAttnInputs,
    ) -> None:
        torch_npu.npu_sparse_flash_attention(
            query=sparse_attn_inputs.query,
            key=gather_inputs.selection_kv_cache.unsqueeze(2),
            value=gather_inputs.selection_kv_cache.unsqueeze(2),
            query_rope=sparse_attn_inputs.query_rope,
            key_rope=gather_inputs.selection_k_rope.unsqueeze(2),
            sparse_indices=sparse_attn_inputs.sparse_indices,
            scale_value=sparse_attn_inputs.scale_value,
            actual_seq_lengths_query=sparse_attn_inputs.actual_seq_lengths_query,
            actual_seq_lengths_kv=sparse_attn_inputs.actual_seq_lengths_kv,
            block_table=gather_inputs.selection_kv_block_table,
            sparse_block_size=sparse_attn_inputs.sparse_block_size,
            layout_query=sparse_attn_inputs.layout_query,
            layout_kv=sparse_attn_inputs.layout_kv,
            sparse_mode=sparse_attn_inputs.sparse_mode,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure serial Indexer + GatherSelectionKvCache + SFA latency."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--reuse-rate", type=float, default=DEFAULT_REUSE_RATE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--full-batch-size", type=int, default=64)
    parser.add_argument("--half-batch-size", type=int, default=32)
    parser.add_argument("--stream-cube-cores", type=int, default=DEFAULT_STREAM_CUBE_CORES)
    parser.add_argument("--stream-vector-cores", type=int, default=DEFAULT_STREAM_VECTOR_CORES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv-name", default=None)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def summarize_metrics(metrics: list[StepMetrics]) -> dict[str, float]:
    indexer = summarize([row.indexer_ms for row in metrics])
    gather = summarize([row.gather_ms for row in metrics])
    sparse_attn = summarize([row.sparse_attn_ms for row in metrics])
    serial = summarize([row.step_ms for row in metrics])
    return {
        "indexer_avg_ms": indexer["avg_ms"],
        "indexer_p50_ms": indexer["p50_ms"],
        "gather_avg_ms": gather["avg_ms"],
        "gather_p50_ms": gather["p50_ms"],
        "sparse_attn_avg_ms": sparse_attn["avg_ms"],
        "sparse_attn_p50_ms": sparse_attn["p50_ms"],
        "serial_sum_avg_ms": serial["avg_ms"],
        "serial_sum_p50_ms": serial["p50_ms"],
        "serial_sum_p90_ms": serial["p90_ms"],
        "serial_sum_p99_ms": serial["p99_ms"],
        "serial_sum_min_ms": serial["min_ms"],
        "serial_sum_max_ms": serial["max_ms"],
    }


def make_runtime(args: argparse.Namespace, batch_size: int, seed_offset: int) -> SerialMeasureRuntime:
    config = BaselineConfig(
        device=args.device,
        batch_size=batch_size,
        seq_len=1,
        kv_max_seq_len=args.max_seq_len,
        index_topk=2048,
        seed=args.seed + seed_offset,
        topk_reuse_rate=args.reuse_rate,
    )
    return SerialMeasureRuntime(config)


def run_case(
    *,
    name: str,
    runtime: BaselineRuntime,
    warmup: int,
    iters: int,
    stream: torch.npu.Stream | None,
) -> list[StepMetrics]:
    def _run_step(step_id: int) -> StepMetrics:
        if stream is None:
            return runtime.run_step(step_id)
        with torch.npu.stream(stream):
            return runtime.run_step(step_id)

    total = warmup + iters
    metrics: list[StepMetrics] = []
    for sample_id in range(total):
        row = _run_step(sample_id)
        if sample_id >= warmup:
            metrics.append(row)
        print(
            f"[{name} sample {sample_id + 1}/{total}] "
            f"indexer={row.indexer_ms:.3f}ms gather={row.gather_ms:.3f}ms "
            f"sfa={row.sparse_attn_ms:.3f}ms serial={row.step_ms:.3f}ms",
            flush=True,
        )
    torch.npu.synchronize()
    return metrics


def row_for_case(
    args: argparse.Namespace,
    name: str,
    batch_size: int,
    limited: bool,
    metrics: list[StepMetrics],
) -> dict[str, object]:
    row: dict[str, object] = {
        "case": name,
        "batch_size": batch_size,
        "max_seq_len": args.max_seq_len,
        "reuse_rate": args.reuse_rate,
        "warmup": args.warmup,
        "iters": args.iters,
        "stream_limited": limited,
        "stream_cube_cores": args.stream_cube_cores if limited else "",
        "stream_vector_cores": args.stream_vector_cores if limited else "",
    }
    row.update(summarize_metrics(metrics))
    return row


def write_csv(rows: list[dict[str, object]], args: argparse.Namespace) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.csv_name is None:
        reuse = f"{args.reuse_rate:g}".replace(".", "p")
        name = (
            f"serial_indexer_gather_sfa_bs{args.full_batch_size}_bs{args.half_batch_size}"
            f"_seq{args.max_seq_len}_reuse{reuse}"
            f"_limit{args.stream_cube_cores}c{args.stream_vector_cores}v.csv"
        )
    else:
        name = args.csv_name
    path = args.out_dir / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    args = parse_args()
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.stream_cube_cores <= 0 or args.stream_vector_cores <= 0:
        raise SystemExit("stream core limits must be positive")
    if not torch.npu.is_available():
        raise SystemExit("NPU not available")

    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.set_device(torch.device(args.device))

    print(
        f"Preparing bs={args.full_batch_size} full-core runtime: "
        f"max_seq={args.max_seq_len}, reuse={args.reuse_rate}",
        flush=True,
    )
    full_runtime = make_runtime(args, args.full_batch_size, seed_offset=0)
    full_metrics = run_case(
        name=f"bs{args.full_batch_size}_full_core",
        runtime=full_runtime,
        warmup=args.warmup,
        iters=args.iters,
        stream=None,
    )

    print(
        f"Preparing bs={args.half_batch_size} half-core runtime: "
        f"stream_limit={args.stream_cube_cores}:{args.stream_vector_cores}",
        flush=True,
    )
    half_runtime = make_runtime(args, args.half_batch_size, seed_offset=97)
    half_stream = torch.npu.Stream()
    torch.npu.set_stream_limit(half_stream, args.stream_cube_cores, args.stream_vector_cores)
    half_metrics = run_case(
        name=f"bs{args.half_batch_size}_half_core",
        runtime=half_runtime,
        warmup=args.warmup,
        iters=args.iters,
        stream=half_stream,
    )

    rows = [
        row_for_case(args, f"bs{args.full_batch_size}_full_core", args.full_batch_size, False, full_metrics),
        row_for_case(args, f"bs{args.half_batch_size}_half_core", args.half_batch_size, True, half_metrics),
    ]
    csv_path = write_csv(rows, args)

    print("\nSummary:")
    for row in rows:
        print(
            f"  {row['case']}: serial={row['serial_sum_avg_ms']:.3f}ms "
            f"(indexer={row['indexer_avg_ms']:.3f}, gather={row['gather_avg_ms']:.3f}, "
            f"sfa={row['sparse_attn_avg_ms']:.3f})"
        )
    print(f"  csv: {csv_path}")


if __name__ == "__main__":
    main()
