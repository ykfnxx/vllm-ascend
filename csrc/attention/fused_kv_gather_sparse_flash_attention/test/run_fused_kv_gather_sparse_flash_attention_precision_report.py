#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Run the fused route-aware SFA precision suite and write JSON/Markdown."""

from __future__ import annotations

import datetime
import json
import os

import torch

from fused_kv_gather_sparse_flash_attention_precision_common import (
    PRECISION_CASES,
    SUPPORTED_DTYPES,
    case_id_string,
    run_precision_case,
)


def render_report(results: list[dict[str, object]]) -> str:
    passed = sum(bool(result["passed"]) for result in results)
    lines = [
        "# FusedKVGatherSparseFlashAttention 精度验证报告",
        "",
        f"生成时间：{datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "Golden 为同一非零 Hot/Host 输入下的 `DsaKVGather + SparseFlashAttention`。",
        "Host KV/RoPE 使用 swapped memory；判定标准为 MERE < dtype threshold 且",
        "MARE < 10 × threshold（FP16=2^-10，BF16=2^-7），并且融合算子写入",
        "install staging 的 Host-miss KV/RoPE 必须与 golden payload 逐元素一致。",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "| ---- | --: |",
        f"| 总用例 | {len(results)} |",
        f"| 通过 | {passed} |",
        f"| 失败 | {len(results) - passed} |",
        f"| 通过率 | {100.0 * passed / len(results):.1f}% |",
        "",
        "## 结果",
        "",
        "| Case | Category | DType | MERE | MARE | MaxAbsErr | StagingErr | Cosine | Result |",
        "| ---- | -------- | ----- | ---: | ---: | --------: | ---------: | -----: | ------ |",
    ]
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"| {case_id_string(result)} | {result['category']} | "
            f"{result['dtype']} | {result['MERE']:.3e} | "
            f"{result['MARE']:.3e} | {result['max_abs_error']:.3e} | "
            f"{result['staging_max_abs_error']:.3e} | "
            f"{result['cosine_similarity']:.10f} | {status} |"
        )
    lines.extend(
        [
            "",
            "## 关键发现",
            "",
            f"- 30 个精度用例中 {passed} 个通过，覆盖 FP16/BF16、BS 1–49、MTP 0/3。",
            "- 覆盖 0%、1%、10%、25%、50%、100% Host miss，验证 Hot/Host 混合路由。",
            "- 覆盖 4K、16K、64K、128K 序列，包含 GLM-5.2 的 BS=12/MTP=3/64K 生产门禁。",
            "- 对每个 Host miss 同时检查 install staging，防止注意力正确但后续 Hot cache 安装错误。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    results: list[dict[str, object]] = []
    device = torch.device("npu:0")
    case_id = 0
    for category, batch_size, mtp_tokens, seq_len, miss_rate in PRECISION_CASES:
        for dtype in SUPPORTED_DTYPES:
            case_id += 1
            result = run_precision_case(
                case_id,
                category,
                batch_size,
                mtp_tokens,
                seq_len,
                miss_rate,
                dtype,
                device,
            )
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"[{status}] {case_id_string(result)} {result['dtype']} "
                f"MERE={result['MERE']:.3e} MARE={result['MARE']:.3e}",
                flush=True,
            )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    stem = "fused_kv_gather_sparse_flash_attention_precision_report"
    json_path = os.path.join(output_dir, f"{stem}.json")
    markdown_path = os.path.join(output_dir, f"{stem}.md")
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)
    with open(markdown_path, "w", encoding="utf-8") as stream:
        stream.write(render_report(results))
    failed = sum(not bool(result["passed"]) for result in results)
    print(f"total={len(results)} passed={len(results) - failed} failed={failed}")
    print(markdown_path)
    print(json_path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
