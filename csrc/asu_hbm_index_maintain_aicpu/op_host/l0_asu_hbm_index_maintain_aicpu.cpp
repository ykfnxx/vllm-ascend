#include "l0_asu_hbm_index_maintain_aicpu.h"

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AsuHbmIndexMaintainAicpu);

const aclTensor* AsuHbmIndexMaintainAicpu(
    const aclTensor* index,
    const aclTensor* slotToIndex,
    const aclTensor* freeSlots,
    const aclTensor* freeHead,
    const aclTensor* reqPoolEntries,
    const aclTensor* lastQuerySlots,
    int64_t reqNum,
    int64_t seed,
    const aclTensor* indexOut,
    const aclTensor* slotToIndexOut,
    const aclTensor* freeSlotsOut,
    const aclTensor* freeHeadOut,
    aclOpExecutor* executor)
{
    L0_DFX(AsuHbmIndexMaintainAicpu,
           index,
           slotToIndex,
           freeSlots,
           freeHead,
           reqPoolEntries,
           lastQuerySlots,
           reqNum,
           seed,
           indexOut,
           slotToIndexOut,
           freeSlotsOut,
           freeHeadOut);

    static internal::AicpuTaskSpace space("AsuHbmIndexMaintainAicpu");
    space.SetRef(0);
    space.SetRef(1);
    space.SetRef(2);
    space.SetRef(3);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        AsuHbmIndexMaintainAicpu,
        OP_ATTR_NAMES({"req_num", "seed"}),
        OP_INPUT(index,
                 slotToIndex,
                 freeSlots,
                 freeHead,
                 reqPoolEntries,
                 lastQuerySlots),
        OP_OUTPUT(indexOut, slotToIndexOut, freeSlotsOut, freeHeadOut),
        OP_ATTR(reqNum, seed));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "AsuHbmIndexMaintainAicpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return indexOut;
}
}  // namespace l0op
