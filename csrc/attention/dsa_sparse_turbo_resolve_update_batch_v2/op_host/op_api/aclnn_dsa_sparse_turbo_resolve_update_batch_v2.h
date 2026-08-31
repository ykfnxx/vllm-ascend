/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef ACLNN_DSA_SPARSE_TURBO_RESOLVE_UPDATE_BATCH_V2_H
#define ACLNN_DSA_SPARSE_TURBO_RESOLVE_UPDATE_BATCH_V2_H

#include "aclnn/aclnn_base.h"

extern "C" {
__attribute__((visibility("default"))) aclnnStatus
aclnnDsaSparseTurboResolveUpdateBatchV2GetWorkspaceSize(
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
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnDsaSparseTurboResolveUpdateBatchV2(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
}

#endif
