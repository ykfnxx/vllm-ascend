/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "aclnn_dsa_offload_resolve_update_batch_v2.h"

extern "C" {
extern aclnnStatus
aclnnInnerDsaOffloadResolveUpdateBatchV2GetWorkspaceSize(
    const aclTensor*,
    const aclTensor*,
    const aclTensor*,
    const aclTensor*,
    const aclTensor*,
    const aclTensor*,
    const aclTensor*,
    const aclTensor*,
    int64_t,
    int64_t,
    int64_t,
    const aclTensor*,
    const aclTensor*,
    uint64_t*,
    aclOpExecutor**);
extern aclnnStatus aclnnInnerDsaOffloadResolveUpdateBatchV2(
    void*, uint64_t, aclOpExecutor*, aclrtStream);

aclnnStatus aclnnDsaOffloadResolveUpdateBatchV2GetWorkspaceSize(
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
    int64_t decodeMode,
    const aclTensor* mappedIndices,
    const aclTensor* gatherMask,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerDsaOffloadResolveUpdateBatchV2GetWorkspaceSize(
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
        decodeMode,
        mappedIndices,
        gatherMask,
        workspaceSize,
        executor);
}

aclnnStatus aclnnDsaOffloadResolveUpdateBatchV2(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    return aclnnInnerDsaOffloadResolveUpdateBatchV2(
        workspace, workspaceSize, executor, stream);
}
}
