#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Profile fused route-aware SFA against SFA and Gather+SFA."""

from __future__ import annotations

import argparse
import datetime
import os
from collections import defaultdict
from typing import Any

import torch

from fused_kv_gather_sparse_flash_attention_profiler_common import (
    PROFILER_SCHEDULE_ACTIVE,
    PROFILER_SCHEDULE_WARMUP,
    build_inputs,
    case_label,
    default_case_file,
    default_trace_root,
    load_cases,
    run_profile,
    verify_accuracy,
)


def _render(results: list[dict[str, Any]], case_file: str, trace_root: str) -> str:
    lines = [
        "# 性能评估结果",
        "",
        f"生成时间：{datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "主标杆为已完成一次 KV materialization 后的 `SparseFlashAttention`；",
        "另附语义等价的 `DsaKvGather + SparseFlashAttention` 完整链路。",
        f"固定 schedule：warmup={PROFILER_SCHEDULE_WARMUP}、active={PROFILER_SCHEDULE_ACTIVE}；",
        "指标为 `op_statistic.csv` 中目标 OP Type 的 `Total Time(us) / 5`。",
        f"用例：`{case_file}`；trace：`{trace_root}`。所有 case 均先完成非零 Hot/Host 精度对比。",
        "",
        "## 性能对比",
        "",
        "| Case | Shape | DType | 融合(us) | SFA(us) | 相对SFA比值 |",
        "| ---- | ----- | ----- | -------: | ------: | ----------: |",
    ]
    for result in results:
        lines.append(
            f"| {result['case']} | {result['shape']} | {result['dtype']} | "
            f"{result['fused_us']:.3f} | {result['sfa_us']:.3f} | "
            f"{result['sfa_us'] / result['fused_us']:.3f} |"
        )

    ratios = [item["sfa_us"] / item["fused_us"] for item in results]
    custom_wins = sum(ratio > 1 for ratio in ratios)
    lines.extend(
        [
            "",
            "## 全量汇总",
            "",
            "| 指标 | 值 |",
            "| ---- | --: |",
            f"| 用例数 | {len(results)} |",
            f"| 平均相对SFA比值（>1 表示融合更快） | {sum(ratios) / len(ratios):.3f} |",
            f"| 融合算子更优 | {custom_wins} |",
            f"| SFA 单算子更优 | {len(results) - custom_wins} |",
            "",
            "### 按数据类型汇总",
            "",
            "| DType | 用例数 | 平均相对SFA比值 | 融合更优 | SFA更优 |",
            "| ----- | -----: | ----------------: | -------: | ------: |",
        ]
    )
    by_dtype: dict[str, list[float]] = defaultdict(list)
    for result in results:
        by_dtype[result["dtype"]].append(result["sfa_us"] / result["fused_us"])
    for dtype, values in sorted(by_dtype.items()):
        wins = sum(value > 1 for value in values)
        lines.append(
            f"| {dtype} | {len(values)} | {sum(values) / len(values):.3f} | "
            f"{wins} | {len(values) - wins} |"
        )

    lines.extend(
        [
            "",
            "## 与未融合完整链路对比",
            "",
            "| Case | Shape | DType | 融合(us) | Gather + SFA(us) | 链路加速比 |",
            "| ---- | ----- | ----- | -------: | -----------------: | ---------: |",
        ]
    )
    for result in results:
        lines.append(
            f"| {result['case']} | {result['shape']} | {result['dtype']} | "
            f"{result['fused_us']:.3f} | {result['chain_us']:.3f} | "
            f"{result['chain_us'] / result['fused_us']:.3f} |"
        )
    avg_sfa = sum(ratios) / len(ratios)
    avg_chain = sum(
        item["chain_us"] / item["fused_us"] for item in results
    ) / len(results)
    lines.extend(
        [
            "",
            "## 简短分析",
            "",
            f"- 相对 SFA 单算子平均比值为 {avg_sfa:.3f}，用于衡量 route 解析和直接 Host/Hot 装载的附加成本。",
            f"- 相对 Gather + SFA 完整链路平均加速比为 {avg_chain:.3f}，用于判断融合是否真正消除 selection-cache 往返。",
            "- BS=12/MTP=3/S=64K 是主要生产门禁；只有完整链路加速且精度通过时才应接入默认路径。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-file", default=default_case_file())
    parser.add_argument("--trace-root", default=default_trace_root())
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--only-case", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases(args.case_file)
    indices = [args.only_case] if args.only_case is not None else range(len(cases))
    results: list[dict[str, Any]] = []
    for index in indices:
        case = cases[index]
        shape, dtype = case_label(case)
        print(f"[case {index}] {shape} {dtype}", flush=True)
        inputs = build_inputs(case, torch.device(args.device))
        verify_accuracy(inputs)
        print("  accuracy: PASS", flush=True)
        timings: dict[str, float] = {}
        for mode in ("fused", "sfa", "chain"):
            timing, csv_path = run_profile(
                mode,
                inputs,
                os.path.join(args.trace_root, mode, f"case_{index:03d}"),
            )
            timings[mode] = timing
            print(f"  {mode}: {timing:.3f} us ({csv_path})", flush=True)
        results.append(
            {
                "case": index,
                "shape": shape,
                "dtype": dtype,
                "fused_us": timings["fused"],
                "sfa_us": timings["sfa"],
                "chain_us": timings["chain"],
            }
        )
        del inputs
        torch.npu.empty_cache()

    report = args.report_md or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fused_kv_gather_sparse_flash_attention_torch_npu_profiler_report.md",
    )
    with open(report, "w", encoding="utf-8") as stream:
        stream.write(
            _render(
                results,
                os.path.abspath(args.case_file),
                os.path.abspath(args.trace_root),
            )
        )
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
