/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "aclnn_dsa_offload_lookup_update_batch.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus
aclnnInnerDsaOffloadLookupUpdateBatchGetWorkspaceSize(
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

extern aclnnStatus aclnnInnerDsaOffloadLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

aclnnStatus aclnnDsaOffloadLookupUpdateBatchGetWorkspaceSize(
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
    return aclnnInnerDsaOffloadLookupUpdateBatchGetWorkspaceSize(
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

aclnnStatus aclnnDsaOffloadLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return aclnnInnerDsaOffloadLookupUpdateBatch(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
