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
    const aclTensor* tokenToHot,
    const aclTensor* hotToToken,
    const aclTensor* lruSlots,
    const aclTensor* stateSeatEpoch,
    const aclTensor* rowToCacheSeat,
    const aclTensor* rowSeatEpoch,
    const aclTensor* queryPositions,
    const aclTensor* queryToRow,
    const aclTensor* queryToLane,
    const aclTensor* queryValidMask,
    const aclTensor* validTopkCounts,
    const aclTensor* seqLens,
    const aclTensor* topkPositions,
    const aclTensor* resolvedHotIndices,
    const aclTensor* missMask,
    const aclTensor* opWorkspace,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerDsaSparseLookupUpdate(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

aclnnStatus aclnnDsaSparseLookupUpdateGetWorkspaceSize(
    const aclTensor* tokenToHot,
    const aclTensor* hotToToken,
    const aclTensor* lruSlots,
    const aclTensor* stateSeatEpoch,
    const aclTensor* rowToCacheSeat,
    const aclTensor* rowSeatEpoch,
    const aclTensor* queryPositions,
    const aclTensor* queryToRow,
    const aclTensor* queryToLane,
    const aclTensor* queryValidMask,
    const aclTensor* validTopkCounts,
    const aclTensor* seqLens,
    const aclTensor* topkPositions,
    const aclTensor* resolvedHotIndices,
    const aclTensor* missMask,
    const aclTensor* opWorkspace,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerDsaSparseLookupUpdateGetWorkspaceSize(
        tokenToHot,
        hotToToken,
        lruSlots,
        stateSeatEpoch,
        rowToCacheSeat,
        rowSeatEpoch,
        queryPositions,
        queryToRow,
        queryToLane,
        queryValidMask,
        validTopkCounts,
        seqLens,
        topkPositions,
        resolvedHotIndices,
        missMask,
        opWorkspace,
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
