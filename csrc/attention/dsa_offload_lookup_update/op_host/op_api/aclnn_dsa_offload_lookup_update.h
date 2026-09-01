/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef ACLNN_DSA_OFFLOAD_LOOKUP_UPDATE_H
#define ACLNN_DSA_OFFLOAD_LOOKUP_UPDATE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnDsaOffloadLookupUpdateGetWorkspaceSize(
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

__attribute__((visibility("default"))) aclnnStatus
aclnnDsaOffloadLookupUpdate(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_DSA_OFFLOAD_LOOKUP_UPDATE_H
