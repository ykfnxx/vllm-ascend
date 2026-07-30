/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "aclnn_asu_hbm_index_lookup.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus
aclnnInnerAsuHbmIndexLookupGetWorkspaceSize(
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

extern aclnnStatus aclnnInnerAsuHbmIndexLookup(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

aclnnStatus aclnnAsuHbmIndexLookupGetWorkspaceSize(
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
    return aclnnInnerAsuHbmIndexLookupGetWorkspaceSize(
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

aclnnStatus aclnnAsuHbmIndexLookup(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return aclnnInnerAsuHbmIndexLookup(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
