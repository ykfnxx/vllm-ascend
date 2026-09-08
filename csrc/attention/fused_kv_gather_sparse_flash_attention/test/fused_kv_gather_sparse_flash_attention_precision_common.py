#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Precision cases and metrics for route-aware fused sparse attention."""

from __future__ import annotations

import math
from typing import Any

import torch

from fused_kv_gather_sparse_flash_attention_profiler_common import (
    ROUTE_KIND_SHIFT,
    SRC_HOST_MISS,
    _fused,
    _gather,
    _sfa,
    build_inputs,
)


PRECISION_CASES = [
    ("Small", 1, 0, 4096, 0.00),
    ("Small", 1, 0, 4096, 0.10),
    ("Small", 4, 0, 4096, 0.10),
    ("Decode", 12, 0, 16384, 0.10),
    ("MTP", 4, 3, 16384, 0.10),
    ("MTP", 12, 3, 16384, 0.10),
    ("Production", 12, 3, 65536, 0.10),
    ("Production", 12, 3, 131072, 0.10),
    ("AllHot", 12, 3, 65536, 0.00),
    ("AllHost", 12, 3, 65536, 1.00),
    ("MissRate", 12, 3, 65536, 0.01),
    ("MissRate", 12, 3, 65536, 0.25),
    ("OddRows", 47, 0, 65536, 0.10),
    ("OddRows", 49, 0, 65536, 0.10),
    ("MissRate", 12, 3, 65536, 0.50),
]

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
THRESHOLDS = {
    torch.float16: 2**-10,
    torch.bfloat16: 2**-7,
}


def make_case(
    batch_size: int,
    mtp_tokens: int,
    seq_len: int,
    miss_rate: float,
    dtype: torch.dtype,
) -> dict[str, Any]:
    rows = batch_size * (mtp_tokens + 1)
    dtype_name = DTYPE_NAMES[dtype]
    return {
        "inputs": [
            {
                "name": "query",
                "type": "tensor",
                "required": True,
                "dtype": dtype_name,
                "shape": [rows, 8, 512],
            },
            {
                "name": "batch_size",
                "type": "attr",
                "required": True,
                "dtype": "int",
                "value": batch_size,
            },
            {
                "name": "mtp_tokens",
                "type": "attr",
                "required": True,
                "dtype": "int",
                "value": mtp_tokens,
            },
            {
                "name": "seq_len",
                "type": "attr",
                "required": True,
                "dtype": "int",
                "value": seq_len,
            },
            {
                "name": "miss_rate",
                "type": "attr",
                "required": True,
                "dtype": "float",
                "value": miss_rate,
            },
        ]
    }


def run_precision_case(
    case_id: int,
    category: str,
    batch_size: int,
    mtp_tokens: int,
    seq_len: int,
    miss_rate: float,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    inputs = build_inputs(
        make_case(batch_size, mtp_tokens, seq_len, miss_rate, dtype), device
    )
    try:
        _gather(inputs)
        golden = _sfa(inputs)
        expected_staging_kv = inputs["selection_kv"].clone()
        expected_staging_rope = inputs["selection_rope"].clone()
        inputs["selection_kv"].zero_()
        inputs["selection_rope"].zero_()
        actual = _fused(inputs)
        torch.npu.synchronize()
        golden_cpu = golden.float().cpu()
        actual_cpu = actual.float().cpu()
        abs_error = (actual_cpu - golden_cpu).abs()
        relative_error = abs_error / (golden_cpu.abs() + 1e-7)
        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.mean().item()
        mare = relative_error.max().item()
        mere = relative_error.mean().item()
        cosine_similarity = torch.nn.functional.cosine_similarity(
            actual_cpu.flatten().unsqueeze(0),
            golden_cpu.flatten().unsqueeze(0),
        ).item()
        miss_mask = (
            torch.bitwise_right_shift(inputs["route_plan"], ROUTE_KIND_SHIFT)
            .eq(SRC_HOST_MISS)
            .reshape(-1)
            .cpu()
        )
        staged_kv = inputs["selection_kv"].view(-1, 512).float().cpu()
        staged_rope = inputs["selection_rope"].view(-1, 64).float().cpu()
        expected_kv = expected_staging_kv.view(-1, 512).float().cpu()
        expected_rope = expected_staging_rope.view(-1, 64).float().cpu()
        if bool(miss_mask.any()):
            staging_max_abs_error = max(
                (staged_kv[miss_mask] - expected_kv[miss_mask]).abs().max().item(),
                (staged_rope[miss_mask] - expected_rope[miss_mask]).abs().max().item(),
            )
        else:
            staging_max_abs_error = max(
                staged_kv.abs().max().item(), staged_rope.abs().max().item()
            )
        threshold = THRESHOLDS[dtype]
        passed = (
            bool(torch.isfinite(actual_cpu).all())
            and golden_cpu.abs().max().item() > 0.0
            and mere < threshold
            and mare < 10 * threshold
            and staging_max_abs_error == 0.0
        )
        return {
            "case_id": case_id,
            "category": category,
            "batch_size": batch_size,
            "mtp_tokens": mtp_tokens,
            "rows": batch_size * (mtp_tokens + 1),
            "seq_len": seq_len,
            "miss_rate": miss_rate,
            "dtype": DTYPE_NAMES[dtype],
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "MARE": mare,
            "MERE": mere,
            "cosine_similarity": cosine_similarity,
            "staging_max_abs_error": staging_max_abs_error,
            "threshold": threshold,
            "passed": passed,
        }
    finally:
        del inputs
        torch.npu.empty_cache()


def case_id_string(result: dict[str, Any]) -> str:
    return (
        f"BS={result['batch_size']},MTP={result['mtp_tokens']},"
        f"S={result['seq_len']},miss={result['miss_rate']:.2f}"
    )
