#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Thirty-case NPU precision regression for fused route-aware SFA."""

import pytest
import torch

from fused_kv_gather_sparse_flash_attention_precision_common import (
    PRECISION_CASES,
    SUPPORTED_DTYPES,
    case_id_string,
    run_precision_case,
)


PARAMETERS = [
    (*case, dtype)
    for case in PRECISION_CASES
    for dtype in SUPPORTED_DTYPES
]


@pytest.mark.parametrize(
    "category,batch_size,mtp_tokens,seq_len,miss_rate,dtype",
    PARAMETERS,
    ids=[
        f"{category}-BS{batch_size}-MTP{mtp_tokens}-S{seq_len}-"
        f"miss{miss_rate}-{str(dtype).removeprefix('torch.')}"
        for category, batch_size, mtp_tokens, seq_len, miss_rate, dtype
        in PARAMETERS
    ],
)
def test_fused_kv_gather_sparse_flash_attention_precision(
    category: str,
    batch_size: int,
    mtp_tokens: int,
    seq_len: int,
    miss_rate: float,
    dtype: torch.dtype,
) -> None:
    result = run_precision_case(
        0,
        category,
        batch_size,
        mtp_tokens,
        seq_len,
        miss_rate,
        dtype,
        torch.device("npu:0"),
    )
    assert result["passed"], (
        f"{case_id_string(result)} dtype={result['dtype']} "
        f"MERE={result['MERE']:.3e} MARE={result['MARE']:.3e}"
    )
