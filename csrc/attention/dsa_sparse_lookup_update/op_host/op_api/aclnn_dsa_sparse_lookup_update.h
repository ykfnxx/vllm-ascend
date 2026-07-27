/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef ACLNN_DSA_SPARSE_LOOKUP_UPDATE_H
#define ACLNN_DSA_SPARSE_LOOKUP_UPDATE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnDsaSparseLookupUpdateGetWorkspaceSize(
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

__attribute__((visibility("default"))) aclnnStatus
aclnnDsaSparseLookupUpdate(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_DSA_SPARSE_LOOKUP_UPDATE_H
