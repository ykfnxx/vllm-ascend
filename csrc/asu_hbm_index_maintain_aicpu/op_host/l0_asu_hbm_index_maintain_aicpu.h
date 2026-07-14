#ifndef L0_ASU_HBM_INDEX_MAINTAIN_AICPU_H
#define L0_ASU_HBM_INDEX_MAINTAIN_AICPU_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* AsuHbmIndexMaintainAicpu(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* lastQuerySlots,
    int64_t reqNum,
    int64_t seed,
    const aclTensor* indexOut,
    const aclTensor* slotToIndexOut,
    const aclTensor* freeSlotsOut,
    const aclTensor* freeHeadOut,
    aclOpExecutor* executor);
}  // namespace l0op

#endif  // L0_ASU_HBM_INDEX_MAINTAIN_AICPU_H
