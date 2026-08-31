/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef ACLNN_DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_H
#define ACLNN_DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnDsaSparseTurboFusedLookupUpdateBatchGetWorkspaceSize(
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

__attribute__((visibility("default"))) aclnnStatus
aclnnDsaSparseTurboFusedLookupUpdateBatch(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_DSA_SPARSE_LOOKUP_UPDATE_TURBO_FUSED_H
