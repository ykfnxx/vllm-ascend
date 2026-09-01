/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "kernel_operator.h"

#include "arch35/dsa_sparse_turbo_fused_lookup_update_batch_simt.h"
#include "dsa_sparse_turbo_fused_lookup_update_batch_common.h"

extern "C" __global__ __aicore__ void dsa_sparse_turbo_fused_lookup_update_batch(
    GM_ADDR index,
    GM_ADDR slot_to_index,
    GM_ADDR free_slots,
    GM_ADDR free_head,
    GM_ADDR request_rows,
    GM_ADDR query_start_loc,
    GM_ADDR query_index,
    GM_ADDR query_positions,
    GM_ADDR verify_starts,
    GM_ADDR tail_starts,
    GM_ADDR mapped_indices,
    GM_ADDR miss_mask,
    GM_ADDR user_workspace,
    GM_ADDR tiling)
{
#ifdef ASCENDC_CPU_DEBUG
    (void)index;
    (void)slot_to_index;
    (void)free_slots;
    (void)free_head;
    (void)request_rows;
    (void)query_start_loc;
    (void)query_index;
    (void)query_positions;
    (void)verify_starts;
    (void)tail_starts;
    (void)mapped_indices;
    (void)miss_mask;
    (void)user_workspace;
    (void)tiling;
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DsaSparseTurboFusedLookupUpdateBatchTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        DsaSparseTurboFusedLookupUpdateBatchTilingData,
        tiling_data,
        tiling);

    __ubuf__ uint32_t
        shared_scratch[DSA_SPARSE_TURBO_FUSED_UB_SCRATCH_WORDS];
    asc_vf_call<
        DsaSparseTurboFusedLookupUpdateBatch::DsaSparseTurboFusedLookupUpdateBatchSimt>(
        dim3(DSA_SPARSE_TURBO_FUSED_SIMT_THREADS),
        reinterpret_cast<__gm__ int32_t*>(index),
        reinterpret_cast<__gm__ int32_t*>(slot_to_index),
        reinterpret_cast<__gm__ int32_t*>(free_slots),
        reinterpret_cast<__gm__ int32_t*>(free_head),
        reinterpret_cast<__gm__ int32_t*>(request_rows),
        reinterpret_cast<__gm__ int32_t*>(query_start_loc),
        reinterpret_cast<__gm__ int32_t*>(query_index),
        reinterpret_cast<__gm__ int32_t*>(query_positions),
        reinterpret_cast<__gm__ int32_t*>(verify_starts),
        reinterpret_cast<__gm__ int32_t*>(tail_starts),
        reinterpret_cast<__gm__ int32_t*>(mapped_indices),
        reinterpret_cast<__gm__ int32_t*>(miss_mask),
        shared_scratch,
        tiling_data.reqNum,
        tiling_data.poolCapacity,
        tiling_data.queryNum,
        tiling_data.indexCapacity,
        tiling_data.isMtp,
        tiling_data.replaceableBase,
        tiling_data.tailBase,
        tiling_data.fallbackSlot,
        tiling_data.stagingBase);
#endif
}
