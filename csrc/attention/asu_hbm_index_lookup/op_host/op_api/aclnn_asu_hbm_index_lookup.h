/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#ifndef ACLNN_ASU_HBM_INDEX_LOOKUP_H
#define ACLNN_ASU_HBM_INDEX_LOOKUP_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnAsuHbmIndexLookupGetWorkspaceSize(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* reqPoolEntries,
    const aclTensor* queryIndex,
    const aclTensor* lookupMask,
    int64_t reqNum,
    const aclTensor* slotOut,
    const aclTensor* missOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnAsuHbmIndexLookup(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_ASU_HBM_INDEX_LOOKUP_H
