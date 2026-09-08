#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Profiler helpers for route-aware fused sparse attention."""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import shutil
import time
from collections.abc import Callable
from typing import Any

import torch
import torch_npu

import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401
# On torch 2.10 importing the extension does not flush the pending
# TORCH_LIBRARY registrations into torch.ops; load_library forces it.
torch.ops.load_library(vllm_ascend.vllm_ascend_C.__file__)
from vllm_ascend.ops.dsa_sparse import (
    dsa_kv_gather,
    fused_kv_gather_sparse_flash_attention,
)


PROFILER_SCHEDULE_WARMUP = 5
PROFILER_SCHEDULE_ACTIVE = 5
BLOCK_SIZE = 128
QUERY_WIDTH = 2048
KV_DIM = 512
ROPE_DIM = 64
QUERY_HEADS = 8
POOL_SLOT_COUNT = 10 * 1024
ROUTE_KIND_SHIFT = 29
SRC_POOL_HIT = 1
SRC_HOST_MISS = 2

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

MODE_OP_TYPES = {
    "fused": {"FusedKvGatherSparseFlashAttention"},
    "sfa": {"SparseFlashAttention"},
    # The serial materialize-then-attend path: dsa_kv_gather is a Python
    # wrapper over per-source AsuKvGather custom ops, so the NPU-side op
    # types measured are AsuKvGather + SparseFlashAttention.
    "chain": {"AsuKvGather", "SparseFlashAttention"},
}


def default_case_file() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fused_kv_gather_sparse_flash_attention_perf_cases.jsonl",
    )


def default_trace_root() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiler_trace")


def load_cases(path: str) -> list[dict[str, Any]]:
    if not path.endswith(".jsonl"):
        raise ValueError("performance cases must use JSONL")
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _specs(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in case["inputs"]}


def build_inputs(case: dict[str, Any], device: torch.device) -> dict[str, Any]:
    specs = _specs(case)
    dtype = DTYPES[specs["query"]["dtype"]]
    batch_size = int(specs["batch_size"]["value"])
    mtp_tokens = int(specs["mtp_tokens"]["value"])
    seq_len = int(specs["seq_len"]["value"])
    miss_rate = float(specs["miss_rate"]["value"])
    queries_per_request = mtp_tokens + 1
    rows = batch_size * queries_per_request
    blocks_per_request = math.ceil(seq_len / BLOCK_SIZE)
    hot_blocks_per_request = math.ceil(POOL_SLOT_COUNT / BLOCK_SIZE)

    torch.manual_seed(20260830)
    query = torch.randn(rows, QUERY_HEADS, KV_DIM, dtype=dtype, device=device)
    query_rope = torch.randn(
        rows, QUERY_HEADS, ROPE_DIM, dtype=dtype, device=device
    )
    query_pool_entries = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).repeat_interleave(queries_per_request)

    hot_block_count = batch_size * hot_blocks_per_request
    hot_kv = torch.randn(
        hot_block_count, BLOCK_SIZE, 1, KV_DIM, dtype=dtype, device=device
    )
    hot_rope = torch.randn(
        hot_block_count, BLOCK_SIZE, 1, ROPE_DIM, dtype=dtype, device=device
    )
    hot_table = torch.arange(
        hot_block_count, dtype=torch.int32, device=device
    ).view(batch_size, hot_blocks_per_request)

    host_block_count = batch_size * blocks_per_request
    host_kv = torch_npu.empty_with_swapped_memory(
        (host_block_count, BLOCK_SIZE, KV_DIM), dtype=dtype, device=device
    )
    host_rope = torch_npu.empty_with_swapped_memory(
        (host_block_count, BLOCK_SIZE, ROPE_DIM), dtype=dtype, device=device
    )
    host_kv.fill_(0.375)
    host_rope.fill_(-0.25)
    host_table = torch.arange(
        host_block_count, dtype=torch.int32, device=device
    ).view(batch_size, blocks_per_request)

    ranks = torch.arange(QUERY_WIDTH, dtype=torch.int32, device=device)
    row_offsets = torch.arange(rows, dtype=torch.int32, device=device).unsqueeze(1)
    token_ids = torch.div(
        (ranks.unsqueeze(0) + 1) * (seq_len - 1),
        QUERY_WIDTH + 1,
        rounding_mode="floor",
    )
    token_ids = torch.remainder(token_ids + row_offsets, seq_len).sort(dim=1).values
    period = max(1, round(1.0 / miss_rate)) if miss_rate > 0 else QUERY_WIDTH + 1
    is_miss = ranks.remainder(period).eq(0).unsqueeze(0).expand(rows, -1)
    source = torch.where(is_miss, token_ids, token_ids.remainder(POOL_SLOT_COUNT))
    route_kind = torch.where(
        is_miss,
        torch.full_like(source, SRC_HOST_MISS),
        torch.full_like(source, SRC_POOL_HIT),
    )
    route_plan = torch.bitwise_or(route_kind << ROUTE_KIND_SHIFT, source).contiguous()
    # The materialized baseline addresses compact selection-cache slots.  The
    # fused path keeps original token positions for SFA validity/threshold
    # handling while route_plan redirects the payload read to Hot or Host.
    selection_indices = (
        ranks.view(1, 1, QUERY_WIDTH).expand(rows, -1, -1).contiguous()
    )
    fused_sparse_indices = token_ids.unsqueeze(1).contiguous()

    selection_blocks_per_row = QUERY_WIDTH // BLOCK_SIZE
    selection_block_count = rows * selection_blocks_per_row
    selection_kv = torch.empty(
        selection_block_count, BLOCK_SIZE, KV_DIM, dtype=dtype, device=device
    )
    selection_rope = torch.empty(
        selection_block_count, BLOCK_SIZE, ROPE_DIM, dtype=dtype, device=device
    )
    selection_physical = torch.arange(
        selection_block_count, dtype=torch.int32, device=device
    ).view(rows, selection_blocks_per_row)
    selection_table = selection_physical[:, :1].expand(
        -1, blocks_per_request
    ).clone()
    selection_table[:, :selection_blocks_per_row].copy_(selection_physical)
    selection_slots = ranks.unsqueeze(0).expand(rows, -1).contiguous()
    actual_query_lengths = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    actual_kv_lengths = torch.full((rows,), seq_len, dtype=torch.int32, device=device)
    scale_value = 1.0 / math.sqrt(KV_DIM + ROPE_DIM)
    torch.npu.synchronize()
    return locals()


def _gather(inputs: dict[str, Any]) -> None:
    dsa_kv_gather(
        inputs["selection_kv"],
        inputs["selection_rope"],
        inputs["selection_table"],
        inputs["hot_kv"].squeeze(2),
        inputs["hot_rope"].squeeze(2),
        inputs["hot_table"],
        inputs["host_kv"],
        inputs["host_rope"],
        inputs["host_table"],
        inputs["query_pool_entries"],
        inputs["query_pool_entries"],
        inputs["selection_slots"],
        inputs["route_plan"],
        BLOCK_SIZE,
    )


def _sfa(inputs: dict[str, Any]) -> torch.Tensor:
    output = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=inputs["query"],
        key=inputs["selection_kv"].unsqueeze(2),
        value=inputs["selection_kv"].unsqueeze(2),
        sparse_indices=inputs["selection_indices"],
        scale_value=inputs["scale_value"],
        sparse_block_size=1,
        block_table=inputs["selection_table"],
        actual_seq_lengths_query=inputs["actual_query_lengths"],
        actual_seq_lengths_kv=inputs["actual_kv_lengths"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["selection_rope"].unsqueeze(2),
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=False,
    )
    return output[0] if isinstance(output, tuple) else output


def _fused(inputs: dict[str, Any]) -> torch.Tensor:
    output, _, _ = fused_kv_gather_sparse_flash_attention(
        inputs["query"],
        inputs["query_rope"],
        (inputs["hot_kv"], inputs["hot_rope"]),
        inputs["hot_table"],
        (inputs["host_kv"], inputs["host_rope"]),
        inputs["host_table"],
        inputs["query_pool_entries"],
        inputs["query_pool_entries"],
        inputs["route_plan"],
        (
            inputs["selection_kv"].unsqueeze(2),
            inputs["selection_rope"].unsqueeze(2),
        ),
        inputs["fused_sparse_indices"],
        inputs["actual_query_lengths"],
        inputs["actual_kv_lengths"],
        scale_value=inputs["scale_value"],
    )
    return output


def forward(mode: str, inputs: dict[str, Any]):
    if mode == "sfa":
        return _sfa(inputs)
    if mode == "chain":
        _gather(inputs)
        return _sfa(inputs)
    if mode == "fused":
        return _fused(inputs)
    raise ValueError(f"unknown mode: {mode}")


def verify_accuracy(inputs: dict[str, Any]) -> None:
    _gather(inputs)
    golden = _sfa(inputs)
    actual = _fused(inputs)
    torch.npu.synchronize()
    if golden.float().abs().max().item() == 0.0:
        raise AssertionError("the SparseFlashAttention golden output is all zero")
    torch.testing.assert_close(actual, golden, rtol=2e-2, atol=2e-2)


def _newest_csv(root: str) -> str:
    paths = glob.glob(os.path.join(root, "**", "op_statistic.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"op_statistic.csv not found under {root}")
    return max(paths, key=os.path.getmtime)


def _column(fieldnames: list[str], predicates: tuple[str, ...]) -> str:
    for field in fieldnames:
        normalized = field.strip().lstrip("\ufeff").replace(" ", "").lower()
        if all(part in normalized for part in predicates):
            return field
    raise KeyError(f"CSV column {predicates} not found in {fieldnames}")


def selected_total_us(csv_path: str, op_types: set[str]) -> float:
    total = 0.0
    seen: set[str] = set()
    with open(csv_path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        op_column = _column(fields, ("op", "type"))
        time_column = _column(fields, ("total", "time", "us"))
        for row in reader:
            op_type = str(row.get(op_column, "")).strip()
            if op_type in op_types:
                total += float(row[time_column])
                seen.add(op_type)
    missing = op_types - seen
    if missing:
        raise RuntimeError(f"missing profiler rows {sorted(missing)} in {csv_path}")
    return total


def run_profile(mode: str, inputs: dict[str, Any], handler_dir: str) -> tuple[float, str]:
    if os.path.isdir(handler_dir):
        shutil.rmtree(handler_dir)
    os.makedirs(handler_dir, exist_ok=True)
    before = time.time()
    experimental = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    )
    callback = torch_npu.profiler.tensorboard_trace_handler(handler_dir)
    steps = PROFILER_SCHEDULE_WARMUP + PROFILER_SCHEDULE_ACTIVE
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=PROFILER_SCHEDULE_WARMUP,
            active=PROFILER_SCHEDULE_ACTIVE,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=callback,
        experimental_config=experimental,
    ) as profiler:
        fn: Callable[[], Any] = lambda: forward(mode, inputs)
        for _ in range(steps):
            fn()
            torch.npu.synchronize()
            profiler.step()
    torch.npu.synchronize()

    deadline = time.time() + 120
    csv_path = ""
    while time.time() < deadline:
        try:
            candidate = _newest_csv(handler_dir)
            if os.path.getmtime(candidate) >= before - 1:
                csv_path = candidate
                break
        except FileNotFoundError:
            pass
        time.sleep(0.2)
    if not csv_path:
        csv_path = _newest_csv(handler_dir)
    total = selected_total_us(csv_path, MODE_OP_TYPES[mode])
    return total / PROFILER_SCHEDULE_ACTIVE, csv_path


def case_label(case: dict[str, Any]) -> tuple[str, str]:
    specs = _specs(case)
    shape = (
        f"BS={specs['batch_size']['value']}, "
        f"MTP={specs['mtp_tokens']['value']}, S={specs['seq_len']['value']}, "
        f"miss={specs['miss_rate']['value']}"
    )
    return shape, str(specs["query"]["dtype"])
