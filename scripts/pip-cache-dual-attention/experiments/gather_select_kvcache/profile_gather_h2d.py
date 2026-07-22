#!/usr/bin/env python3
"""Profile npu_gather_selection_kv_cache: break down CPU/NPU time vs H2D semantics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu

# Reuse perf harness
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_npu_gather_selection_kv_cache_perf import (  # noqa: E402
    HEAD_NUM,
    SEQ_LEN,
    SELECTION_TOPK,
    SELECTION_TOPK_BLOCK_SIZE,
    make_inputs,
    prepare_gather_step,
    random_selection_topk,
    run_gather,
)

try:
    import custom_ops  # noqa: F401
except ImportError as exc:
    raise SystemExit("custom_ops required") from exc


def _parse_op_summary_csv(csv_path: Path) -> list[dict]:
    import csv

    rows = []
    if not csv_path.is_file():
        return rows
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _summarize_prof_dir(prof_dir: Path) -> dict:
    out: dict = {"prof_dir": str(prof_dir), "artifacts": [], "op_summary_top": []}
    if not prof_dir.is_dir():
        return out
    files = sorted(p for p in prof_dir.rglob("*") if p.is_file())
    out["artifacts"] = [str(p.relative_to(prof_dir)) for p in files[:80]]
    # Ascend profiler often emits op_summary under PROF_* tree
    op_csvs = list(prof_dir.rglob("op_summary*.csv"))
    if not op_csvs:
        op_csvs = list(prof_dir.rglob("*op*summary*.csv"))
    for csv_path in op_csvs[:3]:
        rows = _parse_op_summary_csv(csv_path)
        # normalize column names (vary by CANN version)
        def dur_us(r):
            for k in ("Total Time(us)", "total_time_us", "Total Time (us)"):
                if k in r and r[k]:
                    try:
                        return float(r[k])
                    except ValueError:
                        pass
            return 0.0

        def name(r):
            for k in ("Op Name", "OP Name", "Name", "op_name"):
                if k in r and r[k]:
                    return r[k]
            return "?"

        ranked = sorted(rows, key=dur_us, reverse=True)[:25]
        out["op_summary_top"].append(
            {
                "file": str(csv_path.relative_to(prof_dir)),
                "ops": [{"name": name(r), "us": dur_us(r)} for r in ranked if dur_us(r) > 0],
            }
        )
    trace_json = prof_dir / "trace.json"
    if trace_json.is_file():
        out["trace_json_bytes"] = trace_json.stat().st_size
    return out


def profile_once(
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    prof_dir: Path,
    offload: bool,
    iters: int = 5,
) -> dict:
    prof_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ASCEND_PROFILER_OUTPUT"] = str(prof_dir)

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

    n_blocks = (max_seq_len + SELECTION_TOPK_BLOCK_SIZE - 1) // SELECTION_TOPK_BLOCK_SIZE
    init_topk = random_selection_topk(
        np.arange(0, n_blocks, dtype=np.int32), batch_size, SEQ_LEN, HEAD_NUM, SELECTION_TOPK
    )
    inputs = make_inputs(batch_size, max_seq_len, init_topk, device, offload)
    reuse_rate = 0.0

    try:
        prof_cm = torch_npu.profiler.profile(
            activities=activities,
            experimental_config=experimental_config,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
    except TypeError:
        prof_cm = torch_npu.profiler.profile(activities=activities)

    with prof_cm as prof:
        for _ in range(3):
            prepare_gather_step(inputs, batch_size, max_seq_len, reuse_rate)
            run_gather(inputs)
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            prepare_gather_step(inputs, batch_size, max_seq_len, reuse_rate)
            run_gather(inputs)
        end.record()
        end.synchronize()
        kernel_ms = start.elapsed_time(end) / iters
        try:
            prof.step()
        except Exception:
            pass

    trace_path = prof_dir / "trace.json"
    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception:
        pass

    summary = _summarize_prof_dir(prof_dir)
    payload_bytes = batch_size * 1 * 1 * 2048 * 1 * (64 + 512) * 2
    summary.update(
        {
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "offload": offload,
            "profiled_kernel_avg_ms": kernel_ms,
            "gbps_est_payload": (payload_bytes / (1024**3)) / (kernel_ms / 1000.0),
            "payload_bytes": payload_bytes,
            "full_cache_bytes": batch_size * max_seq_len * (64 + 512) * 2,
        }
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="npu:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-seq-len", type=int, default=65536)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "prof_out")
    p.add_argument(
        "--offload-full-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = p.parse_args()

    if not torch.npu.is_available():
        raise SystemExit("NPU not available")

    device = torch.device(args.device)
    torch.npu.set_device(device)

    tag = f"bs{args.batch_size}_seq{args.max_seq_len}_off{int(args.offload_full_cache)}"
    prof_dir = args.out_dir / tag
    result = profile_once(
        args.batch_size,
        args.max_seq_len,
        device,
        prof_dir,
        offload=args.offload_full_cache,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
