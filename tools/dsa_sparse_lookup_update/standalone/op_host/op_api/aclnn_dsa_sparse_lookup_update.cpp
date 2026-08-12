/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "aclnn_dsa_sparse_lookup_update.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus
aclnnInnerDsaSparseLookupUpdateGetWorkspaceSize(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* reqPoolEntries,
    const aclTensor* queryIndex,
    const aclTensor* lookupMask,
    int64_t reqNum,
    const aclTensor* slotOut,
    const aclTensor* missOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerDsaSparseLookupUpdate(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

aclnnStatus aclnnDsaSparseLookupUpdateGetWorkspaceSize(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* reqPoolEntries,
    const aclTensor* queryIndex,
    const aclTensor* lookupMask,
    int64_t reqNum,
    const aclTensor* slotOut,
    const aclTensor* missOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerDsaSparseLookupUpdateGetWorkspaceSize(
        index,
        slotToIndex,
        freeSlots,
        freeHead,
        reqPoolEntries,
        queryIndex,
        lookupMask,
        reqNum,
        slotOut,
        missOut,
        workspaceSize,
        executor);
}

aclnnStatus aclnnDsaSparseLookupUpdate(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return aclnnInnerDsaSparseLookupUpdate(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
