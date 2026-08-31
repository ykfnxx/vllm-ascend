/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "aclnn_dsa_sparse_turbo_fused_prefetch_lookup_update_batch.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus
aclnnInnerDsaSparseTurboFusedPrefetchLookupUpdateBatchGetWorkspaceSize(
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
    const aclTensor* destinationSlots,
    const aclTensor* missMask,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerDsaSparseTurboFusedPrefetchLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

aclnnStatus aclnnDsaSparseTurboFusedPrefetchLookupUpdateBatchGetWorkspaceSize(
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
    const aclTensor* destinationSlots,
    const aclTensor* missMask,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerDsaSparseTurboFusedPrefetchLookupUpdateBatchGetWorkspaceSize(
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
        destinationSlots,
        missMask,
        workspaceSize,
        executor);
}

aclnnStatus aclnnDsaSparseTurboFusedPrefetchLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return aclnnInnerDsaSparseTurboFusedPrefetchLookupUpdateBatch(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
