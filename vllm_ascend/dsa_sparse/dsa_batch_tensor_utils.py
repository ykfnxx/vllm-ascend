"""DSA batch 级元数据的无状态张量构造工具。

本文件只放“把 Python list / request rows 转成本轮 forward 张量”的纯工具：
padding、排序、HBM/DRAM block table gather，以及 sparse attention indices
宽度计算。它不认识 DSASparseV1，也不修改
请求/资源状态。

这样拆分后，dsa_sparse.py 负责调度 DSA 生命周期和 layer hook，本文件负责
稳定的张量物化规则。后续如果要继续拆 DSAModelForwardMeta 或
DSAForwardSparseDecodeBatch，可以继续沿着这个边界迁移。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

IntRow = Sequence[int] | torch.Tensor


def _int_row_len(row: IntRow | None) -> int:
    if row is None:
        return 0
    if torch.is_tensor(row):
        return int(row.numel())
    return len(row)


def build_padded_int_tensor(
        rows: list[IntRow],
        *,
        dtype: torch.dtype,
        device: torch.device,
        pad_value: int = -1) -> torch.Tensor:
    width = max((_int_row_len(row) for row in rows), default=0)
    if width == 0:
        return torch.full((len(rows), 0),
                          int(pad_value),
                          dtype=dtype,
                          device=device)
    if any(torch.is_tensor(row) for row in rows):
        output = torch.full((len(rows), width),
                            int(pad_value),
                            dtype=dtype,
                            device=device)
        for row_idx, row in enumerate(rows):
            row_len = _int_row_len(row)
            if row_len <= 0:
                continue
            if torch.is_tensor(row):
                row_tensor = row.reshape(-1).to(device=device, dtype=dtype)
            else:
                row_tensor = torch.tensor([int(item) for item in row],
                                          dtype=dtype,
                                          device=device)
            output[row_idx, :min(row_len, width)].copy_(
                row_tensor[:min(row_len, width)])
        return output
    padded_rows = [
        [int(item) for item in row] + [int(pad_value)] * (width - len(row))
        for row in rows
    ]
    return torch.tensor(padded_rows, dtype=dtype, device=device)


def build_int_tensor(
        values: list[int],
        *,
        dtype: torch.dtype,
        device: torch.device) -> torch.Tensor:
    return torch.tensor([int(item) for item in values],
                        dtype=dtype,
                        device=device)


def sort_decode_rows_by_batch_index(
        batch_row_indices: list[int],
        *row_aligned_lists: list,
) -> tuple:
    """Keep decode rows in the model-forward query row order."""
    if len(batch_row_indices) <= 1:
        return (batch_row_indices, *row_aligned_lists)

    order = sorted(
        range(len(batch_row_indices)),
        key=lambda row: int(batch_row_indices[row]),
    )
    if order == list(range(len(order))):
        return (batch_row_indices, *row_aligned_lists)

    return (
        [int(batch_row_indices[row]) for row in order],
        *([values[row] for row in order] for values in row_aligned_lists),
    )


def build_hbm_block_table_tensor(
        full_block_table: torch.Tensor | None,
        row_indices: list[int],
        slot_counts: list[int],
        *,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
        pad_value: int = 0) -> torch.Tensor:
    """Gather compact HBM block-table rows for paged DSA IO.

    vLLM reserves HBM block id 0 as the null block, so the IO-facing table uses
    0 for padding/invalid entries. The table maps request-local resident
    positions to physical HBM cache blocks; offsets are computed inside the IO
    operator.
    """
    row_count = len(row_indices)
    if row_count != len(slot_counts):
        raise RuntimeError(
            "DSA HBM block table build got mismatched rows and lengths: "
            f"{row_count} vs {len(slot_counts)}")
    slot_counts = [max(0, int(count)) for count in slot_counts]
    max_slots = max(slot_counts, default=0)
    if row_count == 0 or max_slots <= 0:
        return torch.full((row_count, 0),
                          int(pad_value),
                          dtype=dtype,
                          device=device)
    block_size = int(block_size)
    if block_size <= 0:
        return torch.full((row_count, 0),
                          int(pad_value),
                          dtype=dtype,
                          device=device)
    if full_block_table is None:
        raise RuntimeError(
            "DSA sparse decode requires full-cache HBM block table metadata")
    if full_block_table.device != device:
        raise RuntimeError(
            "DSA full-cache HBM block table must already be on the target "
            f"device; got {full_block_table.device} vs {device}")

    max_blocks = (max_slots + block_size - 1) // block_size
    row_tensor = build_int_tensor(row_indices,
                                  dtype=torch.long,
                                  device=device)
    block_ids = full_block_table.index_select(0, row_tensor)
    if block_ids.dtype != torch.long:
        block_ids = block_ids.to(torch.long)
    if int(block_ids.shape[1]) < max_blocks:
        pad_cols = max_blocks - int(block_ids.shape[1])
        block_ids = torch.cat((
            block_ids,
            torch.full((row_count, pad_cols),
                       int(pad_value),
                       dtype=torch.long,
                       device=device),
        ), dim=1)
    return block_ids[:, :max_blocks].to(dtype=dtype)


def compute_sparse_attention_indices_width(
    *,
    budget_slot_count: int,
) -> int:
    """Return the original Indexer TopK width after resident-slot mapping."""
    return max(0, int(budget_slot_count))
