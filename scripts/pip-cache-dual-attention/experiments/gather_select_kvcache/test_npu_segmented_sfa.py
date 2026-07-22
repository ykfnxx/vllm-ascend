#!/usr/bin/env python3
# coding=utf-8
"""Validate segmented SFA driven by KVSelect hit/miss outputs.

Reference path:
  KVSelect + KVGather -> one full SFA over selection slots 0..K-1

Segmented path:
  KVSelect -> hit SFA on pre-gather HBM selection cache
           -> KVGather for misses
           -> miss SFA on newly gathered selection slots
           -> LSE merge of the two partial attention outputs
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

_PRELOAD_OLD_OP_API = os.getenv("DMP_PRELOAD_OLD_OP_API")
_OLD_OP_API_HANDLE = None
if _PRELOAD_OLD_OP_API:
    _OLD_OP_API_HANDLE = ctypes.CDLL(
        _PRELOAD_OLD_OP_API, mode=ctypes.RTLD_GLOBAL
    )

import torch
import torch_npu  # noqa: F401

from test_npu_gather_selection_kv_cache_perf import (
    HEAD_NUM,
    K_ROPE_DIM,
    KV_CACHE_DIM,
    SELECTION_TOPK,
    SELECTION_TOPK_BLOCK_SIZE,
    SEQ_LEN,
    make_inputs,
    random_selection_topk,
)
from test_npu_kv_select_gather_correctness import (
    clone_inputs,
    make_next_topk,
    run_split,
    set_topk,
)
from test_npu_kv_select_gather_perf import make_workspace, run_kv_gather, run_kv_select

_REPO_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from dual_attention import da_attention_merge, run_sparse_flash_attention  # noqa: E402


QUERY_HEAD_NUM = 128
QUERY_SEED_OFFSET = 1009
ALLCLOSE_RTOL = 0.02
ALLCLOSE_ATOL = 0.001

TensorMap = dict[str, torch.Tensor]


@dataclass(frozen=True)
class SegmentedSfaCase:
    batch_size: int
    max_seq_len: int
    reuse_rate: float
    device: torch.device
    seed: int
    offload: bool


@dataclass(frozen=True)
class HitMissCounts:
    hit: int
    miss: int


@dataclass(frozen=True)
class SelectionSfaContext:
    query: torch.Tensor
    query_rope: torch.Tensor
    inputs: TensorMap
    actual_seq: torch.Tensor


@dataclass(frozen=True)
class SegmentedSfaOutput:
    attention: torch.Tensor
    counts: HitMissCounts


@dataclass(frozen=True)
class ValidationResult:
    case: SegmentedSfaCase
    max_diff: float
    counts: HitMissCounts


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def make_query(batch_size: int, device: torch.device, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    token_count = batch_size * SEQ_LEN
    query = torch.tensor(
        rng.normal(0.0, 0.1, (token_count, QUERY_HEAD_NUM, KV_CACHE_DIM)),
        dtype=dtype,
        device=device,
    )
    query_rope = torch.tensor(
        rng.normal(0.0, 0.1, (token_count, QUERY_HEAD_NUM, K_ROPE_DIM)),
        dtype=dtype,
        device=device,
    )
    return query, query_rope


def compute_selection_actual_seq(inputs: TensorMap) -> torch.Tensor:
    max_selected = SELECTION_TOPK * SELECTION_TOPK_BLOCK_SIZE
    full_actual = inputs["full_kv_actual_seq"].view(-1)
    max_actual = torch.full(
        full_actual.shape,
        max_selected,
        dtype=full_actual.dtype,
        device=full_actual.device,
    )
    return torch.minimum(full_actual, max_actual).to(torch.int32)


def build_full_sparse_indices(actual_seq: torch.Tensor) -> torch.Tensor:
    rows = actual_seq.numel()
    cols = torch.arange(SELECTION_TOPK, dtype=torch.int32, device=actual_seq.device).view(1, 1, -1)
    cols = cols.expand(rows, HEAD_NUM, -1)
    invalid = torch.full(cols.shape, -1, dtype=cols.dtype, device=cols.device)
    return torch.where(cols < actual_seq.view(rows, 1, 1), cols, invalid).contiguous()


def compact_valid_indices(tensor: torch.Tensor) -> tuple[torch.Tensor | None, int]:
    rows = tensor.numel() // SELECTION_TOPK
    packed = tensor.reshape(rows, HEAD_NUM, SELECTION_TOPK)
    counts = (packed >= 0).sum(dim=-1)
    max_count = int(counts.max().item())
    if max_count <= 0:
        return None, 0
    return packed[..., :max_count].contiguous(), int(counts.sum().item())


def run_selection_sfa(
    context: SelectionSfaContext,
    sparse_indices: torch.Tensor,
    *,
    softmax_max_out: torch.Tensor | None = None,
    softmax_sum_out: torch.Tensor | None = None,
) -> torch.Tensor:
    return run_sparse_flash_attention(
        context.query,
        context.inputs["selection_kv_cache"].unsqueeze(2),
        context.inputs["selection_kv_cache"].unsqueeze(2),
        sparse_indices,
        float(1.0 / math.sqrt(KV_CACHE_DIM)),
        query_rope=context.query_rope,
        key_rope=context.inputs["selection_k_rope"].unsqueeze(2),
        block_table=context.inputs["selection_kv_block_table"],
        actual_seq_lengths_query=torch.arange(
            1,
            context.query.shape[0] + 1,
            dtype=torch.int32,
            device=context.query.device,
        ),
        actual_seq_lengths_kv=context.actual_seq,
        sparse_block_size=SELECTION_TOPK_BLOCK_SIZE,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        softmax_max_out=softmax_max_out,
        softmax_sum_out=softmax_sum_out,
    )


def run_segmented_sfa(
    context: SelectionSfaContext,
    workspace: TensorMap,
) -> SegmentedSfaOutput:
    run_kv_select(context.inputs, workspace)

    hit_indices, hit_total = compact_valid_indices(workspace["hit_sparse_indices"])
    miss_indices, miss_total = compact_valid_indices(workspace["miss_insert_indices"])
    counts = HitMissCounts(hit=hit_total, miss=miss_total)

    lse_shape = (context.query.shape[0], context.query.shape[1])
    neg_inf = torch.finfo(torch.float32).min
    hit_max = torch.full(lse_shape, neg_inf, dtype=torch.float32, device=context.query.device)
    hit_sum = torch.zeros(lse_shape, dtype=torch.float32, device=context.query.device)
    miss_max = torch.full(lse_shape, neg_inf, dtype=torch.float32, device=context.query.device)
    miss_sum = torch.zeros(lse_shape, dtype=torch.float32, device=context.query.device)

    if hit_indices is None:
        hit_out = context.query.new_zeros(context.query.shape)
    else:
        hit_out = run_selection_sfa(
            context,
            hit_indices,
            softmax_max_out=hit_max,
            softmax_sum_out=hit_sum,
        )

    run_kv_gather(context.inputs, workspace)
    if not torch.equal(workspace["selection_kv_actual_seq"].view(-1), context.actual_seq.view(-1)):
        raise AssertionError("KVGather selection_kv_actual_seq differs from expected selected length")

    if miss_indices is None:
        final = hit_out
    elif hit_indices is None:
        final = run_selection_sfa(
            context,
            miss_indices,
        )
    else:
        miss_out = run_selection_sfa(
            context,
            miss_indices,
            softmax_max_out=miss_max,
            softmax_sum_out=miss_sum,
        )
        final = da_attention_merge(hit_out, miss_out, hit_max, hit_sum, miss_max, miss_sum)

    return SegmentedSfaOutput(attention=final, counts=counts)


def assert_attention_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    counts: HitMissCounts,
) -> float:
    max_diff = (actual.float() - expected.float()).abs().max().item()
    if not torch.allclose(actual.float(), expected.float(), rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL):
        raise AssertionError(f"segmented SFA mismatch: max_abs_diff={max_diff:.6f}, counts={counts}")
    return max_diff


def run_case(case: SegmentedSfaCase) -> ValidationResult:
    rng = np.random.default_rng(case.seed)
    np.random.seed(case.seed)
    n_blocks = (case.max_seq_len + SELECTION_TOPK_BLOCK_SIZE - 1) // SELECTION_TOPK_BLOCK_SIZE
    topk = random_selection_topk(
        np.arange(0, n_blocks, dtype=np.int32), case.batch_size, SEQ_LEN, HEAD_NUM, SELECTION_TOPK
    )
    base_inputs = make_inputs(case.batch_size, case.max_seq_len, topk, case.device, offload_full_cache=case.offload)

    prime_workspace = make_workspace(base_inputs, case.batch_size, case.device)
    run_split(base_inputs, prime_workspace)

    next_topk = make_next_topk(topk, case.batch_size, case.max_seq_len, case.reuse_rate, rng)
    set_topk(base_inputs, next_topk)

    reference_inputs = clone_inputs(base_inputs)
    segmented_inputs = clone_inputs(base_inputs)
    workspace = make_workspace(segmented_inputs, case.batch_size, case.device)
    query, query_rope = make_query(
        case.batch_size,
        case.device,
        reference_inputs["selection_kv_cache"].dtype,
        case.seed + QUERY_SEED_OFFSET,
    )
    expected_actual_seq = compute_selection_actual_seq(base_inputs)

    reference_workspace = make_workspace(reference_inputs, case.batch_size, case.device)
    reference_actual = run_split(reference_inputs, reference_workspace)
    reference_context = SelectionSfaContext(
        query=query,
        query_rope=query_rope,
        inputs=reference_inputs,
        actual_seq=reference_actual.view(-1),
    )
    full_out = run_selection_sfa(
        reference_context,
        build_full_sparse_indices(reference_context.actual_seq),
    )

    segmented_context = SelectionSfaContext(
        query=query,
        query_rope=query_rope,
        inputs=segmented_inputs,
        actual_seq=expected_actual_seq,
    )

    segmented_out = run_segmented_sfa(segmented_context, workspace)
    torch.npu.synchronize()

    max_diff = assert_attention_close(segmented_out.attention, full_out, segmented_out.counts)
    return ValidationResult(case=case, max_diff=max_diff, counts=segmented_out.counts)


def build_cases(args: argparse.Namespace, device: torch.device) -> list[SegmentedSfaCase]:
    if not args.sweep:
        return [
            SegmentedSfaCase(
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                reuse_rate=args.reuse_rate,
                device=device,
                seed=args.seed,
                offload=args.offload_full_cache,
            )
        ]

    batch_sizes = parse_int_list(args.batch_sizes)
    max_seq_lens = parse_int_list(args.max_seq_lens)
    reuse_rates = parse_float_list(args.reuse_rates)
    offloads = [False, True] if args.sweep_offload_both else [args.offload_full_cache]
    cases: list[SegmentedSfaCase] = []
    case_idx = 0
    for offload in offloads:
        for batch_size in batch_sizes:
            for max_seq_len in max_seq_lens:
                for reuse_rate in reuse_rates:
                    cases.append(
                        SegmentedSfaCase(
                            batch_size=batch_size,
                            max_seq_len=max_seq_len,
                            reuse_rate=reuse_rate,
                            device=device,
                            seed=args.seed + case_idx,
                            offload=offload,
                        )
                    )
                    case_idx += 1
    return cases


def format_result(result: ValidationResult) -> str:
    case = result.case
    return (
        f"segmented sfa ok: bs={case.batch_size} max_seq={case.max_seq_len} "
        f"reuse={case.reuse_rate:.2f} offload={case.offload} "
        f"hit={result.counts.hit} miss={result.counts.miss} "
        f"max_abs_diff={result.max_diff:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate segmented SFA against split gather plus full SFA.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--reuse-rate", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--offload-full-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--max-seq-lens", default="4096,8192,16384")
    parser.add_argument("--reuse-rates", default="0.0,0.5,0.9,1.0")
    parser.add_argument("--sweep-offload-both", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.npu.is_available():
        raise SystemExit("NPU not available")

    import custom_ops  # noqa: F401

    os.environ.setdefault("USE_CUSTOM_SFA", "1")
    device = torch.device(args.device)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.set_device(device)

    results = [run_case(case) for case in build_cases(args, device)]
    for result in results:
        print(format_result(result))
    if len(results) > 1:
        max_diff = max(result.max_diff for result in results)
        print(f"segmented sfa sweep ok: cases={len(results)} max_abs_diff={max_diff:.6f}")


if __name__ == "__main__":
    main()
