/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "aclnn_dsa_sparse_lookup_update_batch.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus
aclnnInnerDsaSparseLookupUpdateBatchGetWorkspaceSize(
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

extern aclnnStatus aclnnInnerDsaSparseLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

aclnnStatus aclnnDsaSparseLookupUpdateBatchGetWorkspaceSize(
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
    return aclnnInnerDsaSparseLookupUpdateBatchGetWorkspaceSize(
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

aclnnStatus aclnnDsaSparseLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return aclnnInnerDsaSparseLookupUpdateBatch(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
