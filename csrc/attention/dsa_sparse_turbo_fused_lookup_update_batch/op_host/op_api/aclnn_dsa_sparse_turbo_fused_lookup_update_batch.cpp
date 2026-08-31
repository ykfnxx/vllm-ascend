/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "aclnn_dsa_sparse_turbo_fused_lookup_update_batch.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus
aclnnInnerDsaSparseTurboFusedLookupUpdateBatchGetWorkspaceSize(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* requestRows,
    const aclTensor* queryStartLoc,
    const aclTensor* queryIndex,
    const aclTensor* queryPositions,
    const aclTensor* verifyStarts,
    int64_t reqNum,
    int64_t blockSize,
    int64_t isMtp,
    const aclTensor* mappedIndices,
    const aclTensor* missMask,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerDsaSparseTurboFusedLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

aclnnStatus aclnnDsaSparseTurboFusedLookupUpdateBatchGetWorkspaceSize(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* requestRows,
    const aclTensor* queryStartLoc,
    const aclTensor* queryIndex,
    const aclTensor* queryPositions,
    const aclTensor* verifyStarts,
    int64_t reqNum,
    int64_t blockSize,
    int64_t isMtp,
    const aclTensor* mappedIndices,
    const aclTensor* missMask,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerDsaSparseTurboFusedLookupUpdateBatchGetWorkspaceSize(
        index,
        slotToIndex,
        freeSlots,
        freeHead,
        requestRows,
        queryStartLoc,
        queryIndex,
        queryPositions,
        verifyStarts,
        reqNum,
        blockSize,
        isMtp,
        mappedIndices,
        missMask,
        workspaceSize,
        executor);
}

aclnnStatus aclnnDsaSparseTurboFusedLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return aclnnInnerDsaSparseTurboFusedLookupUpdateBatch(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
