#!/usr/bin/env python3
"""msprof-based batch vs turbo_fused A5 comparison over the CSV cases.

For every case in hit_rate_cases.csv it launches one msprof session wrapping
msprof_case.py, then reads the resulting op_summary_*.csv and extracts the
steady-state Task Duration(us) of DsaSparseLookupUpdateBatch vs
DsaSparseTurboLookupUpdateBatch.  This bypasses the aclnn launch overhead of
wall-clock loops and attributes pure kernel time (methodology from the coarse
operator: wall-clock is masked by aclnn launch overhead; op_summary is the
authority).

Run inside the container on an A5 node (source set_env.bash first):
    python3 run_msprof_compare.py --cases 5 --n 50
    python3 run_msprof_compare.py --case b016_c128k_h090 --n 50
    python3 run_msprof_compare.py --all --n 50
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "hit_rate_cases.csv"
OP_BATCH = "DsaSparseTurboLookupUpdateBatch"
OP_TURBO = "DsaSparseTurboFusedLookupUpdateBatch"


def _find_op_summary(out_dir: Path):
    for pattern in ("PROF_*", "PROF_*/*", "PROF_*/PROF_*"):
        hits = sorted(glob.glob(str(out_dir / pattern / "mindstudio_profiler_output" / "op_summary_*.csv")))
        if hits:
            return hits[0]
    return None


def _steady_duration_us(op_summary_path, op_name, tail_n=5):
    rows = []
    with open(op_summary_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Op Name") == op_name:
                try:
                    rows.append(float(row["Task Duration(us)"]))
                except (KeyError, ValueError):
                    continue
    if not rows:
        return None
    return statistics.mean(rows[-tail_n:])


def run_case(name, batch, capacity, hit, n, out_root):
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    app = (
        f"python3 {HERE / 'msprof_case.py'} "
        f"--batch {batch} --capacity {capacity} --hit {hit} --n {n}"
    )
    cmd = [
        "msprof",
        f"--output={out_dir}",
        f"--application={app}",
        "--ai-core=on",
        "--aic-mode=task-based",
        "--task-time=on",
    ]
    subprocess.run(cmd, check=False, capture_output=True, text=True)

    op_summary = _find_op_summary(out_dir)
    if op_summary is None:
        print(f"{name}: op_summary not found under {out_dir}")
        return None
    batch_us = _steady_duration_us(op_summary, OP_BATCH, tail_n=n)
    turbo_us = _steady_duration_us(op_summary, OP_TURBO, tail_n=n)
    if batch_us is None or turbo_us is None:
        print(f"{name}: missing op rows (batch={batch_us}, turbo={turbo_us}) in {op_summary}")
        return None
    return {
        "name": name, "batch": batch, "capacity_k": capacity // 1024,
        "hit_rate": hit, "batch_us": batch_us, "turbo_us": turbo_us,
        "speedup": batch_us / turbo_us if turbo_us > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every CSV case")
    ap.add_argument("--cases", type=int, help="run first N CSV cases")
    ap.add_argument("--case", type=str, help="run a single named case")
    ap.add_argument("--n", type=int, default=50, help="timed iterations per op")
    ap.add_argument("--out", type=str, default=str(HERE / "msprof_out"))
    args = ap.parse_args()

    with open(CSV) as f:
        cases = list(csv.DictReader(f))

    if args.case:
        cases = [c for c in cases if c["name"] == args.case]
        if not cases:
            print(f"case {args.case!r} not found in {CSV}")
            sys.exit(1)
    elif args.cases:
        cases = cases[: args.cases]
    elif not args.all:
        cases = cases[:5]  # default: a quick 5-case sweep

    out_root = Path(args.out)
    results = []
    print(f"{'case':42s} {'bs':>3s} {'cap':>6s} {'hit%':>6s} "
          f"{'batch(us)':>10s} {'turbo(us)':>10s} {'speedup':>8s}")
    print("-" * 100)
    for c in cases:
        name = c["name"]
        batch = int(c["batch"])
        capacity = int(c["index_capacity"])
        hit = float(c["hit_rate"])
        r = run_case(name, batch, capacity, hit, args.n, out_root)
        if r is None:
            continue
        results.append(r)
        print(f"{r['name']:42s} {r['batch']:3d} {r['capacity_k']:6d} {r['hit_rate']:6.4f} "
              f"{r['batch_us']:10.3f} {r['turbo_us']:10.3f} {r['speedup']:7.3f}x")

    if results:
        speedups = [r["speedup"] for r in results]
        print("=" * 100)
        print(f"Summary ({len(results)} cases): min={min(speedups):.3f}x "
              f"median={statistics.median(speedups):.3f}x "
              f"mean={statistics.mean(speedups):.3f}x max={max(speedups):.3f}x")


if __name__ == "__main__":
    main()
