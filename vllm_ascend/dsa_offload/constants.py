# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

INDEX_CAPACITY = 128 * 1024
RESIDENT_SLOTS = 8 * 1024
REPLACEABLE_SLOTS = 2 * 1024
LOOKUP_SLOTS = RESIDENT_SLOTS + REPLACEABLE_SLOTS
QUERY_WIDTH = 2 * 1024
FREE_HEAD_STRIDE = 16
FALLBACK_SENTINEL = LOOKUP_SLOTS

# route_plan encoding consumed by the fused_kv_gather_sparse_flash_attention
# kernel: kind in bits [31:29], source position in bits [28:0].
ROUTE_KIND_SHIFT = 29
ROUTE_KIND_INVALID = 0
ROUTE_KIND_POOL_HIT = 1
ROUTE_KIND_HOST_MISS = 2
