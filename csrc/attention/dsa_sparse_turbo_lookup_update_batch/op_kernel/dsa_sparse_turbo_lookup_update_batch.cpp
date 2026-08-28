/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "kernel_operator.h"

#include "arch35/dsa_sparse_turbo_lookup_update_batch_simt.h"
#include "dsa_sparse_turbo_lookup_update_batch_common.h"

extern "C" __global__ __aicore__ void dsa_sparse_turbo_lookup_update_batch(
    GM_ADDR index,
    GM_ADDR slot_to_index,
    GM_ADDR free_slots,
    GM_ADDR free_head,
    GM_ADDR req_pool_entries,
    GM_ADDR query_start_loc,
    GM_ADDR query_index,
    GM_ADDR lookup_mask,
    GM_ADDR slot_out,
    GM_ADDR miss_out,
    GM_ADDR user_workspace,
    GM_ADDR tiling)
{
#ifdef ASCENDC_CPU_DEBUG
    (void)index;
    (void)slot_to_index;
    (void)free_slots;
    (void)free_head;
    (void)req_pool_entries;
    (void)query_start_loc;
    (void)query_index;
    (void)lookup_mask;
    (void)slot_out;
    (void)miss_out;
    (void)user_workspace;
    (void)tiling;
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DsaSparseTurboLookupUpdateBatchTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        DsaSparseTurboLookupUpdateBatchTilingData,
        tiling_data,
        tiling);

    __ubuf__ uint32_t
        shared_scratch[DSA_SPARSE_TURBO_UB_SCRATCH_WORDS];
    asc_vf_call<
        DsaSparseTurboLookupUpdateBatch::DsaSparseTurboLookupUpdateBatchSimt>(
        dim3(DSA_SPARSE_TURBO_SIMT_THREADS),
        reinterpret_cast<__gm__ int32_t*>(index),
        reinterpret_cast<__gm__ int32_t*>(slot_to_index),
        reinterpret_cast<__gm__ int32_t*>(free_slots),
        reinterpret_cast<__gm__ int32_t*>(free_head),
        reinterpret_cast<__gm__ int32_t*>(req_pool_entries),
        reinterpret_cast<__gm__ int32_t*>(query_start_loc),
        reinterpret_cast<__gm__ int32_t*>(query_index),
        reinterpret_cast<__gm__ int32_t*>(lookup_mask),
        reinterpret_cast<__gm__ int32_t*>(slot_out),
        reinterpret_cast<__gm__ int32_t*>(miss_out),
        shared_scratch,
        tiling_data.reqNum,
        tiling_data.poolCapacity,
        tiling_data.queryNum,
        tiling_data.indexCapacity);
#endif
}
