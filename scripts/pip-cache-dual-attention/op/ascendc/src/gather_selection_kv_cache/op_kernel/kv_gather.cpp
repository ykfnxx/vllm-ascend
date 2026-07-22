/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "kernel_operator.h"
#include "gather_selection_kv_cache_split_bs_reuse_vec.h"

using namespace AscendC;
using namespace GatherSelectionKvCacheNs;

extern "C" __global__ __aicore__ void kv_gather(
    GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache, GM_ADDR selection_kv_block_table,
    GM_ADDR selection_kv_block_status, GM_ADDR miss_topk_indices, GM_ADDR miss_insert_indices, GM_ADDR full_k_rope,
    GM_ADDR full_kv_cache, GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq, GM_ADDR full_q_actual_seq,
    GM_ADDR hit_actual_seq, GM_ADDR miss_actual_seq, GM_ADDR miss_count, GM_ADDR hit_count,
    GM_ADDR selection_status_empty, GM_ADDR selection_k_rope_out, GM_ADDR selection_kv_cache_out,
    GM_ADDR selection_kv_block_table_out, GM_ADDR selection_kv_block_status_out, GM_ADDR selection_kv_actual_seq,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GatherSelectionKvCacheSplitBsReuseVec<DTYPE_FULL_K_ROPE> op(&pipe, &tilingData);
    op.InitKVGather(
        selection_k_rope, selection_kv_cache, selection_kv_block_table, selection_kv_block_status,
        miss_topk_indices, miss_insert_indices, full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq,
        full_q_actual_seq, hit_actual_seq, miss_actual_seq, miss_count, hit_count, selection_status_empty,
        selection_kv_actual_seq);
    op.ProcessKVGather();
}
