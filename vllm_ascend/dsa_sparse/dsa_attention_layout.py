"""DSA attention layout 的只读抽取工具。

本文件只负责从 vLLM/vLLM-Ascend 的 attention metadata 中读取并抽取
DSA 需要的 attention layout 信息，例如 query 累计长度、dense/indexer
positions，以及 full/MLA HBM block table。这里不会修改原始
attention metadata，不持有请求生命周期状态，也不触碰 DRAM/HBM 资源池；
调用方每轮 forward 按需切片后再写入 DSAModelForwardMeta。

拆出这个文件的目的，是把 dsa_sparse.py 中和框架 metadata 适配相关的
只读字段解析逻辑独立出来。后续如果 vLLM-Ascend 的 DeepSeek/MLA
metadata 字段再变，优先在这里维护“从哪里读 layout”，避免把字段判断
散落到 scheduler、worker 和 layer hook。
"""

from __future__ import annotations

import torch


def materialize_forward_int_list(values: torch.Tensor) -> list[int]:
    return [
        int(item)
        for item in values.detach().to("cpu").reshape(-1).tolist()
    ]


def has_query_layout_metadata(attn_metadata) -> bool:
    return torch.is_tensor(attn_metadata.cum_query_lens)


def is_indexer_only_attention_metadata(attn_metadata) -> bool:
    return (
        torch.is_tensor(attn_metadata.indexer_block_tables)
        and not torch.is_tensor(attn_metadata.full_block_tables))


def select_forward_shared_metadata(attn_metadata):
    """Pick the full/MLA metadata object that owns forward query layout."""
    if not isinstance(attn_metadata, dict):
        return attn_metadata

    for metadata in attn_metadata.values():
        if (has_query_layout_metadata(metadata)
                and not is_indexer_only_attention_metadata(metadata)):
            return metadata
    raise RuntimeError(
        "DSA requires full/MLA attention metadata for query layout")


def slice_position_row(values: torch.Tensor, start: int,
                       length: int) -> torch.Tensor:
    """Slice scheduled query positions without forcing tensor rows to CPU.

    Decode hot path only needs the current request row. Keep the slice as a
    tensor and let batch tensor construction copy it directly to the target
    device.
    """
    start = int(start)
    end = start + max(0, int(length))
    return values[start:end].reshape(-1)


def materialize_query_position_metadata(attn_metadata) -> dict[str, object]:
    """Return lazy query-position metadata for the current model forward.

    Keep position tensors lazy: build_dsa_meta() only slices/materializes them
    for scheduled request rows, so long prefill chunks do not pay a full
    positions D2H/list conversion just to dump full blocks.
    """
    cum_query_lens = attn_metadata.cum_query_lens
    if not torch.is_tensor(cum_query_lens):
        raise RuntimeError(
            "DSA requires full/MLA attention metadata with cum_query_lens "
            f"for forward-level query layout, got {type(attn_metadata).__name__}")
    indexer_positions = attn_metadata.indexer_positions
    resident_positions = attn_metadata.resident_positions
    if not torch.is_tensor(indexer_positions):
        raise RuntimeError("DSA requires indexer_positions in SFA metadata")
    if not torch.is_tensor(resident_positions):
        raise RuntimeError("DSA requires resident_positions in SFA metadata")
    return {
        "cum_query_lens": materialize_forward_int_list(cum_query_lens),
        "indexer_positions": indexer_positions,
        "resident_positions": resident_positions,
    }


def resolve_full_block_table_tensor(attn_metadata) -> torch.Tensor | None:
    """Find the full/MLA HBM block-table tensor from attention metadata."""
    if isinstance(attn_metadata, dict):
        for metadata in attn_metadata.values():
            full_block_table = metadata.full_block_tables
            if full_block_table is not None:
                return full_block_table
        return None
    return attn_metadata.full_block_tables
