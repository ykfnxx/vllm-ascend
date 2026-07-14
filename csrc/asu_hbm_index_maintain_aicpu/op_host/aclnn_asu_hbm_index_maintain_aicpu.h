#ifndef ACLNN_ASU_HBM_INDEX_MAINTAIN_AICPU_H
#define ACLNN_ASU_HBM_INDEX_MAINTAIN_AICPU_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnAsuHbmIndexMaintainAicpuGetWorkspaceSize(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* lastQuerySlots,
    int64_t reqNum,
    int64_t seed,
    aclTensor* indexOut,
    aclTensor* slotToIndexOut,
    aclTensor* freeSlotsOut,
    aclTensor* freeHeadOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnAsuHbmIndexMaintainAicpu(void* workspace,
                              uint64_t workspaceSize,
                              aclOpExecutor* executor,
                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_ASU_HBM_INDEX_MAINTAIN_AICPU_H
