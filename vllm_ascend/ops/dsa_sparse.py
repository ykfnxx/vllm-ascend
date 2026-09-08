# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Python wrappers for the DSA sparse-attention custom operations.

The fused kernel (`fused_kv_gather_sparse_flash_attention`) reads the Hot
(on-die HBM) and Host (swapped memory) KV caches directly, dispatching each
selected token to one of the two sources according to ``route_plan`` without
materializing a selection cache.

For benchmark purposes, ``dsa_kv_gather`` provides the serial
"materialize the selection cache first, then run ordinary SFA" path built on
top of the existing single-source ``asu_kv_gather`` custom op: the hot-route
and host-route tokens are gathered into the compact selection cache by two
calls, then ordinary SparseFlashAttention consumes it.
"""

from __future__ import annotations

import torch

from vllm_ascend.dsa_offload.ops import (
    asu_kv_gather,
    fused_kv_gather_sparse_flash_attention,
)

ROUTE_KIND_SHIFT = 29
ROUTE_KIND_MASK = 0x7
SOURCE_MASK = (1 << ROUTE_KIND_SHIFT) - 1
SRC_POOL_HIT = 1
SRC_HOST_MISS = 2


def _route_plan_split(route_plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    route = (route_plan >> ROUTE_KIND_SHIFT) & ROUTE_KIND_MASK
    source = route_plan & SOURCE_MASK
    return route, source


def dsa_kv_gather(
    destination_kv_cache: torch.Tensor,
    destination_k_rope: torch.Tensor,
    destination_block_table: torch.Tensor,
    hot_kv: torch.Tensor,
    hot_rope: torch.Tensor,
    hot_block_table: torch.Tensor,
    host_kv: torch.Tensor,
    host_rope: torch.Tensor,
    host_block_table: torch.Tensor,
    request_rows: torch.Tensor,
    hot_source_rows: torch.Tensor,
    destination_slots: torch.Tensor,
    route_plan: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the selection cache from the Hot and Host sources.

    This mirrors the serial baseline of the fused kernel: each (row, token)
    pair is copied from either the Hot cache (slot-indexed) or the Host cache
    (token-indexed) into its compact selection slot, then ordinary
    SparseFlashAttention reads the selection cache.
    """
    route, source = _route_plan_split(route_plan)
    # asu_kv_gather uses the dense [requests, query_count] pair space.
    slots = destination_slots
    src = source
    # asu_kv_gather requires an int32 miss mask (nonzero = route this pair).
    hot_mask = (route == SRC_POOL_HIT).to(torch.int32)
    host_mask = (route == SRC_HOST_MISS).to(torch.int32)

    # Hot-route tokens: source pool row = hot_source_rows, token = hot slot.
    if bool(hot_mask.any()):
        asu_kv_gather(
            destination_kv_cache,
            destination_k_rope,
            destination_block_table,
            hot_kv,
            hot_rope,
            hot_block_table,
            hot_source_rows,
            src,
            slots,
            hot_mask,
            block_size,
        )
    # Host-route tokens: source pool row = request_rows, token = host token id.
    if bool(host_mask.any()):
        asu_kv_gather(
            destination_kv_cache,
            destination_k_rope,
            destination_block_table,
            host_kv,
            host_rope,
            host_block_table,
            request_rows,
            src,
            slots,
            host_mask,
            block_size,
        )
    return destination_kv_cache, destination_k_rope


__all__ = [
    "dsa_kv_gather",
    "fused_kv_gather_sparse_flash_attention",
    "asu_kv_gather",
]
