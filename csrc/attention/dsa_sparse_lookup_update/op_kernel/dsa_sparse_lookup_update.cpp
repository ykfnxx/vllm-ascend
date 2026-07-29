/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "kernel_operator.h"

#include "arch35/dsa_sparse_lookup_update_simt.h"
#include "dsa_sparse_lookup_update_common.h"

extern "C" __global__ __aicore__ void dsa_sparse_lookup_update(
    GM_ADDR index,
    GM_ADDR slot_to_index,
    GM_ADDR free_slots,
    GM_ADDR free_head,
    GM_ADDR req_pool_entries,
    GM_ADDR query_index,
    GM_ADDR lookup_mask,
    GM_ADDR slot_out,
    GM_ADDR miss_out,
    GM_ADDR system_workspace,
    GM_ADDR tiling)
{
#ifdef ASCENDC_CPU_DEBUG
    (void)index;
    (void)slot_to_index;
    (void)free_slots;
    (void)free_head;
    (void)req_pool_entries;
    (void)query_index;
    (void)lookup_mask;
    (void)slot_out;
    (void)miss_out;
    (void)system_workspace;
    (void)tiling;
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DsaSparseLookupUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        DsaSparseLookupUpdateTilingData,
        tiling_data,
        tiling);

    const uint32_t first_req_id =
        static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aiv_count =
        static_cast<uint32_t>(AscendC::GetBlockNum());
    if (first_req_id >= tiling_data.reqNum ||
        aiv_count == 0U) {
        return;
    }

    for (uint32_t req_id = first_req_id;
         req_id < tiling_data.reqNum;
         req_id += aiv_count) {
        asc_vf_call<
            DsaSparseLookupUpdate::DsaSparseLookupUpdateSimt>(
            dim3(DSA_SPARSE_SIMT_THREADS),
            reinterpret_cast<__gm__ int32_t*>(index),
            reinterpret_cast<__gm__ int32_t*>(slot_to_index),
            reinterpret_cast<__gm__ int32_t*>(free_slots),
            reinterpret_cast<__gm__ int32_t*>(free_head),
            reinterpret_cast<__gm__ int32_t*>(req_pool_entries),
            reinterpret_cast<__gm__ int32_t*>(query_index),
            reinterpret_cast<__gm__ int32_t*>(lookup_mask),
            reinterpret_cast<__gm__ int32_t*>(slot_out),
            reinterpret_cast<__gm__ int32_t*>(miss_out),
            reinterpret_cast<__gm__ int32_t*>(
                system_workspace),
            req_id,
            tiling_data.poolCapacity,
            tiling_data.workspaceStride);
    }
#endif
}
