/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file gather_selection_kv_cache.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "gather_selection_kv_cache_split_bs_reuse.h"
#include "gather_selection_kv_cache_split_bs_reuse_vec.h"

using namespace AscendC;
using namespace GatherSelectionKvCacheNs;
extern "C" __global__ __aicore__ void gather_selection_kv_cache(
    GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache, GM_ADDR selection_kv_block_table,
    GM_ADDR selection_kv_block_status, GM_ADDR req_pool_entries, GM_ADDR selection_topk_indices, GM_ADDR full_k_rope,
    GM_ADDR full_kv_cache, GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq, GM_ADDR row_modes,
    GM_ADDR budget_lengths, GM_ADDR tail_valid_token_counts, GM_ADDR resident_tail_starts,
    GM_ADDR query_position_rows, GM_ADDR attention_indices_out, GM_ADDR selection_k_rope_out,
    GM_ADDR selection_kv_cache_out, GM_ADDR selection_kv_block_table_out, GM_ADDR selection_kv_block_status_out,
    GM_ADDR attention_indices_out_out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(3)) {
        GatherSelectionKvCacheSplitBsReuseVec<DTYPE_FULL_K_ROPE> op(&pipe, &tilingData);
        op.InitRowMode(
            selection_k_rope, selection_kv_cache, selection_kv_block_table, selection_kv_block_status,
            req_pool_entries, selection_topk_indices, full_k_rope, full_kv_cache, full_kv_block_table,
            full_kv_actual_seq, row_modes, budget_lengths, tail_valid_token_counts, resident_tail_starts,
            query_position_rows, attention_indices_out);
        op.Process();
    }
}
