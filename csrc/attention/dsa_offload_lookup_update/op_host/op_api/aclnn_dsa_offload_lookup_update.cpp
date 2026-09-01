/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "aclnn_dsa_offload_lookup_update.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus
aclnnInnerDsaOffloadLookupUpdateGetWorkspaceSize(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* requestRows,
    const aclTensor* queryStartLoc,
    const aclTensor* queryPositions,
    const aclTensor* semanticTopk,
    int64_t reqNum,
    int64_t blockSize,
    int64_t tailBase,
    int64_t fallbackSlot,
    int64_t stagingBase,
    int64_t decodeMode,
    const aclTensor* mappedIndices,
    const aclTensor* missMask,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerDsaOffloadLookupUpdate(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

aclnnStatus aclnnDsaOffloadLookupUpdateGetWorkspaceSize(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* requestRows,
    const aclTensor* queryStartLoc,
    const aclTensor* queryPositions,
    const aclTensor* semanticTopk,
    int64_t reqNum,
    int64_t blockSize,
    int64_t tailBase,
    int64_t fallbackSlot,
    int64_t stagingBase,
    int64_t decodeMode,
    const aclTensor* mappedIndices,
    const aclTensor* missMask,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerDsaOffloadLookupUpdateGetWorkspaceSize(
        index,
        slotToIndex,
        freeSlots,
        freeHead,
        requestRows,
        queryStartLoc,
        queryPositions,
        semanticTopk,
        reqNum,
        blockSize,
        tailBase,
        fallbackSlot,
        stagingBase,
        decodeMode,
        mappedIndices,
        missMask,
        workspaceSize,
        executor);
}

aclnnStatus aclnnDsaOffloadLookupUpdate(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return aclnnInnerDsaOffloadLookupUpdate(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
