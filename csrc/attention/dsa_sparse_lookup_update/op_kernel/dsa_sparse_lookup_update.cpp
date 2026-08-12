/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "kernel_operator.h"

#include "arch35/dsa_sparse_lookup_update_simt.h"
#include "dsa_sparse_lookup_update_common.h"

extern "C" __vector__ __global__ __aicore__ void dsa_sparse_lookup_update(
    GM_ADDR index,
    GM_ADDR slot_to_index,
    GM_ADDR free_slots,
    GM_ADDR free_head,
    GM_ADDR req_pool_entries,
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
    (void)query_index;
    (void)lookup_mask;
    (void)slot_out;
    (void)miss_out;
    (void)user_workspace;
    (void)tiling;
#else
    REGISTER_TILING_DEFAULT(DsaSparseLookupUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        DsaSparseLookupUpdateTilingData,
        tiling_data,
        tiling);

    // Launch one VF per active AIV. The VF distributes complete requests
    // across the physical AIV grid so the per-request cooperative SIMT work
    // and UB scratch lifetime stay inside one thread block.
    __ubuf__ uint32_t
        shared_scratch[DSA_SPARSE_UB_SCRATCH_WORDS];
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
        shared_scratch,
        tiling_data.reqNum,
        tiling_data.poolCapacity);
#endif
}
