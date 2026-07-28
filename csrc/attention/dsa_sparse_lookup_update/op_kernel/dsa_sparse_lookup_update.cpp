/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "kernel_operator.h"

#include "arch35/dsa_sparse_lookup_update_simt.h"
#include "dsa_sparse_lookup_update_common.h"

extern "C" __global__ __aicore__ void dsa_sparse_lookup_update(
    GM_ADDR token_to_hot,
    GM_ADDR hot_to_token,
    GM_ADDR lru_slots,
    GM_ADDR query_positions,
    GM_ADDR query_to_req_idx,
    GM_ADDR query_to_lane,
    GM_ADDR query_valid_mask,
    GM_ADDR valid_topk_counts,
    GM_ADDR seq_lens,
    GM_ADDR topk_positions,
    GM_ADDR resolved_hot_indices,
    GM_ADDR miss_mask,
    GM_ADDR op_workspace,
    GM_ADDR system_workspace,
    GM_ADDR tiling)
{
    (void)system_workspace;
#ifdef ASCENDC_CPU_DEBUG
    (void)token_to_hot;
    (void)hot_to_token;
    (void)lru_slots;
    (void)query_positions;
    (void)query_to_req_idx;
    (void)query_to_lane;
    (void)query_valid_mask;
    (void)valid_topk_counts;
    (void)seq_lens;
    (void)topk_positions;
    (void)resolved_hot_indices;
    (void)miss_mask;
    (void)op_workspace;
    (void)tiling;
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DsaSparseLookupUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        DsaSparseLookupUpdateTilingData,
        tiling_data,
        tiling);

    const uint32_t first_request_index =
        static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aiv_count =
        static_cast<uint32_t>(AscendC::GetBlockNum());
    if (first_request_index >= tiling_data.requestCapacity ||
        aiv_count == 0U) {
        return;
    }

    for (uint32_t request_index = first_request_index;
         request_index < tiling_data.requestCapacity;
         request_index += aiv_count) {
        asc_vf_call<
            DsaSparseLookupUpdate::DsaSparseLookupUpdateSimt>(
            dim3(DSA_SPARSE_SIMT_THREADS),
            reinterpret_cast<__gm__ int32_t*>(token_to_hot),
            reinterpret_cast<__gm__ int32_t*>(hot_to_token),
            reinterpret_cast<__gm__ int32_t*>(lru_slots),
            reinterpret_cast<__gm__ int32_t*>(query_positions),
            reinterpret_cast<__gm__ int32_t*>(query_to_req_idx),
            reinterpret_cast<__gm__ int32_t*>(query_to_lane),
            reinterpret_cast<__gm__ uint8_t*>(
                query_valid_mask),
            reinterpret_cast<__gm__ int32_t*>(
                valid_topk_counts),
            reinterpret_cast<__gm__ int32_t*>(seq_lens),
            reinterpret_cast<__gm__ int32_t*>(topk_positions),
            reinterpret_cast<__gm__ int32_t*>(
                resolved_hot_indices),
            reinterpret_cast<__gm__ uint8_t*>(miss_mask),
            reinterpret_cast<__gm__ int32_t*>(op_workspace),
            request_index,
            tiling_data.tokenPositionCapacity,
            tiling_data.evictableSlotCount,
            tiling_data.queryCapacity,
            tiling_data.queryLaneCapacity,
            tiling_data.topkCount,
            tiling_data.workspaceStride);
    }
#endif
}
