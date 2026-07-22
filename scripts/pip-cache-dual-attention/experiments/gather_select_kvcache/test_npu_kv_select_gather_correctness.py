#!/usr/bin/env python3
# coding=utf-8
"""Validate KVSelect+KVGather directly against the full KV cache payload."""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch_npu  # noqa: F401

from test_npu_gather_selection_kv_cache_perf import (
    FULL_BLOCK_SIZE,
    HEAD_NUM,
    SELECTION_BLOCK_SIZE,
    SELECTION_TOPK,
    SELECTION_TOPK_BLOCK_SIZE,
    SEQ_LEN,
    make_inputs,
    random_selection_topk,
)
from test_npu_kv_select_gather_perf import make_workspace, run_kv_gather, run_kv_select

MUTATED_INPUTS = {
    "selection_k_rope",
    "selection_kv_cache",
    "selection_kv_block_table",
    "selection_kv_block_status",
}


def clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() if name in MUTATED_INPUTS else tensor for name, tensor in inputs.items()}


def make_next_topk(
    cur_topk: np.ndarray,
    batch_size: int,
    max_seq_len: int,
    reuse_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    reuse_rate = max(0.0, min(1.0, reuse_rate))
    n_blocks = (max_seq_len + SELECTION_TOPK_BLOCK_SIZE - 1) // SELECTION_TOPK_BLOCK_SIZE
    all_ids = np.arange(0, n_blocks, dtype=np.int32)
    n_replace = int(round(SELECTION_TOPK * (1.0 - reuse_rate)))
    if n_replace <= 0:
        return cur_topk.copy()
    if n_replace >= SELECTION_TOPK:
        return random_selection_topk(all_ids, batch_size, SEQ_LEN, HEAD_NUM, SELECTION_TOPK)

    out = cur_topk.copy()
    for b in range(batch_size):
        for s in range(SEQ_LEN):
            for h in range(HEAD_NUM):
                slots = rng.choice(SELECTION_TOPK, size=n_replace, replace=False)
                kept = np.delete(out[b, s, h], slots)
                usable = np.asarray(list(set(all_ids) - set(kept)), dtype=np.int32)
                if usable.size > 0:
                    out[b, s, h, slots] = rng.choice(usable, size=n_replace, replace=False)
    return out


def set_topk(inputs: dict[str, torch.Tensor], topk: np.ndarray) -> None:
    inputs["selection_topk_indices"].copy_(
        torch.from_numpy(topk).to(device=inputs["selection_topk_indices"].device)
    )


def clear_status(inputs: dict[str, torch.Tensor]) -> None:
    inputs["selection_kv_block_status"].fill_(-1)


def assert_equal_tensor(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if torch.equal(lhs, rhs):
        return
    diff = (lhs != rhs).reshape(-1)
    first = int(diff.nonzero()[0].item()) if diff.any() else -1
    raise AssertionError(f"{name} mismatch, first_flat_index={first}")


def run_split(inputs: dict[str, torch.Tensor], ws: dict[str, torch.Tensor]) -> torch.Tensor:
    run_kv_select(inputs, ws)
    run_kv_gather(inputs, ws)
    torch.npu.synchronize()
    return ws["selection_kv_actual_seq"]


def assert_split_state(
    inputs: dict[str, torch.Tensor],
    ws: dict[str, torch.Tensor],
    actual_seq: torch.Tensor,
    case_name: str,
) -> None:
    topk = inputs["selection_topk_indices"].reshape(-1, SELECTION_TOPK).detach().cpu()
    status = inputs["selection_kv_block_status"].reshape(-1, SELECTION_TOPK + 1).detach().cpu()
    rows = topk.shape[0]

    # The package smoke test generates exactly K valid token ids per row.
    expected_count = (topk >= 0).sum(dim=-1).to(torch.int32)
    if not torch.all(expected_count == SELECTION_TOPK):
        raise AssertionError(f"{case_name}: smoke test requires {SELECTION_TOPK} valid top-k ids")

    actual_seq = actual_seq.reshape(-1).detach().cpu()
    assert_equal_tensor(f"{case_name}:actual_seq", expected_count, actual_seq)
    assert_equal_tensor(f"{case_name}:status_actual_seq", actual_seq, status[:, -1])

    stored_ids = status[:, :SELECTION_TOPK]
    if torch.any(stored_ids < 0):
        raise AssertionError(f"{case_name}: selection status contains an empty slot")
    assert_equal_tensor(
        f"{case_name}:selected_ids",
        torch.sort(topk, dim=-1).values,
        torch.sort(stored_ids, dim=-1).values,
    )

    hit_count = ws["hit_count"].reshape(-1).detach().cpu()
    miss_count = ws["miss_count"].reshape(-1).detach().cpu()
    assert_equal_tensor(f"{case_name}:hit_miss_count", expected_count, hit_count + miss_count)

    slots = torch.arange(SELECTION_TOPK, dtype=torch.int64).view(1, -1).expand(rows, -1)
    selection_table = inputs["selection_kv_block_table"].reshape(rows, -1).detach().cpu().to(torch.int64)
    selection_blocks = selection_table.gather(1, slots // SELECTION_BLOCK_SIZE)
    selection_offsets = slots % SELECTION_BLOCK_SIZE

    batch_ids = (
        torch.arange(rows, dtype=torch.int64) // (SEQ_LEN * HEAD_NUM)
    ).view(-1, 1).expand(rows, SELECTION_TOPK)
    full_table = inputs["full_kv_block_table"].detach().cpu().to(torch.int64)
    full_blocks = full_table[batch_ids, stored_ids.to(torch.int64) // FULL_BLOCK_SIZE]
    full_offsets = stored_ids.to(torch.int64) % FULL_BLOCK_SIZE

    for selection_name, full_name in (
        ("selection_k_rope", "full_k_rope"),
        ("selection_kv_cache", "full_kv_cache"),
    ):
        selection_cache = inputs[selection_name].detach().cpu()
        full_cache = inputs[full_name].detach().cpu()
        selected_payload = selection_cache[selection_blocks, selection_offsets]
        expected_payload = full_cache[full_blocks, full_offsets]
        assert_equal_tensor(f"{case_name}:{selection_name}", expected_payload, selected_payload)


def compare_once(
    split_inputs: dict[str, torch.Tensor],
    split_ws: dict[str, torch.Tensor],
    case_name: str,
) -> None:
    split_actual = run_split(split_inputs, split_ws)
    assert_split_state(split_inputs, split_ws, split_actual, case_name)


def run_case(batch_size: int, max_seq_len: int, reuse_rate: float, device: torch.device, seed: int) -> None:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    n_blocks = (max_seq_len + SELECTION_TOPK_BLOCK_SIZE - 1) // SELECTION_TOPK_BLOCK_SIZE
    topk = random_selection_topk(
        np.arange(0, n_blocks, dtype=np.int32), batch_size, SEQ_LEN, HEAD_NUM, SELECTION_TOPK
    )
    base_inputs = make_inputs(batch_size, max_seq_len, topk, device, offload_full_cache=False)
    split_inputs = clone_inputs(base_inputs)
    split_ws = make_workspace(split_inputs, batch_size, device)

    compare_once(split_inputs, split_ws, "cold")

    if reuse_rate <= 0.0:
        clear_status(split_inputs)
    topk = make_next_topk(topk, batch_size, max_seq_len, reuse_rate, rng)
    set_topk(split_inputs, topk)
    compare_once(split_inputs, split_ws, f"reuse_{reuse_rate:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate split KVSelect+KVGather against full KV payload.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--reuse-rate", type=float, action="append", default=None)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.npu.is_available():
        raise SystemExit("NPU not available")

    device = torch.device(args.device)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.set_device(device)

    reuse_rates = args.reuse_rate if args.reuse_rate is not None else [0.0, 0.9, 1.0]
    for idx, reuse_rate in enumerate(reuse_rates):
        run_case(args.batch_size, args.max_seq_len, reuse_rate, device, args.seed + idx)

    print(
        f"split correctness ok: bs={args.batch_size} max_seq={args.max_seq_len} "
        f"reuse={','.join(f'{rate:.2f}' for rate in reuse_rates)}"
    )


if __name__ == "__main__":
    main()
