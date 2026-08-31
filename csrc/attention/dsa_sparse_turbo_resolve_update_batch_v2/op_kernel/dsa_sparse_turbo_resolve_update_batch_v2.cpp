/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "kernel_operator.h"
#include "arch35/dsa_sparse_turbo_resolve_update_batch_v2_simt.h"
#include "dsa_sparse_turbo_resolve_update_batch_v2_common.h"

extern "C" __global__ __aicore__ void
dsa_sparse_turbo_resolve_update_batch_v2(
    GM_ADDR index,
    GM_ADDR slotToIndex,
    GM_ADDR freeSlots,
    GM_ADDR freeHead,
    GM_ADDR requestRows,
    GM_ADDR queryStartLoc,
    GM_ADDR queryPositions,
    GM_ADDR semanticTopk,
    GM_ADDR mappedIndices,
    GM_ADDR gatherMask,
    GM_ADDR userWorkspace,
    GM_ADDR tiling)
{
    (void)userWorkspace;
#ifdef ASCENDC_CPU_DEBUG
    (void)index;
    (void)slotToIndex;
    (void)freeSlots;
    (void)freeHead;
    (void)requestRows;
    (void)queryStartLoc;
    (void)queryPositions;
    (void)semanticTopk;
    (void)mappedIndices;
    (void)gatherMask;
    (void)tiling;
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DsaSparseTurboResolveUpdateBatchV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(
        DsaSparseTurboResolveUpdateBatchV2TilingData, data, tiling);
    __ubuf__ uint32_t scratch[
        DSA_TURBO_RESOLVE_V2_UB_SCRATCH_WORDS];
    asc_vf_call<DsaSparseTurboResolveUpdateBatchV2::ResolveUpdateSimt>(
        dim3(DSA_TURBO_RESOLVE_V2_SIMT_THREADS),
        reinterpret_cast<__gm__ int32_t*>(index),
        reinterpret_cast<__gm__ int32_t*>(slotToIndex),
        reinterpret_cast<__gm__ int32_t*>(freeSlots),
        reinterpret_cast<__gm__ int32_t*>(freeHead),
        reinterpret_cast<__gm__ int32_t*>(requestRows),
        reinterpret_cast<__gm__ int32_t*>(queryStartLoc),
        reinterpret_cast<__gm__ int32_t*>(queryPositions),
        reinterpret_cast<__gm__ int32_t*>(semanticTopk),
        reinterpret_cast<__gm__ int32_t*>(mappedIndices),
        reinterpret_cast<__gm__ int32_t*>(gatherMask),
        scratch,
        data.reqNum,
        data.poolCapacity,
        data.queryNum,
        data.blockSize,
        data.decodeMode);
#endif
}
