/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "aclnn_dsa_sparse_turbo_prefetch_lookup_update_batch.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus
aclnnInnerDsaSparseTurboPrefetchLookupUpdateBatchGetWorkspaceSize(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* reqPoolEntries,
    const aclTensor* queryStartLoc,
    const aclTensor* queryIndex,
    const aclTensor* lookupMask,
    int64_t reqNum,
    const aclTensor* slotOut,
    const aclTensor* missOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerDsaSparseTurboPrefetchLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

aclnnStatus aclnnDsaSparseTurboPrefetchLookupUpdateBatchGetWorkspaceSize(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* reqPoolEntries,
    const aclTensor* queryStartLoc,
    const aclTensor* queryIndex,
    const aclTensor* lookupMask,
    int64_t reqNum,
    const aclTensor* slotOut,
    const aclTensor* missOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerDsaSparseTurboPrefetchLookupUpdateBatchGetWorkspaceSize(
        index,
        slotToIndex,
        freeSlots,
        freeHead,
        reqPoolEntries,
        queryStartLoc,
        queryIndex,
        lookupMask,
        reqNum,
        slotOut,
        missOut,
        workspaceSize,
        executor);
}

aclnnStatus aclnnDsaSparseTurboPrefetchLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return aclnnInnerDsaSparseTurboPrefetchLookupUpdateBatch(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
