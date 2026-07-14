#include "aclnn_asu_hbm_index_maintain_aicpu.h"

#include "l0_asu_hbm_index_maintain_aicpu.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAsuHbmIndexMaintainAicpuGetWorkspaceSize(
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
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnAsuHbmIndexMaintainAicpu,
                   DFX_IN(index,
                          slotToIndex,
                          freeSlots,
                          freeHead,
                          lastQuerySlots,
                          reqNum,
                          seed),
                   DFX_OUT(indexOut,
                           slotToIndexOut,
                           freeSlotsOut,
                           freeHeadOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,
              ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto output = l0op::AsuHbmIndexMaintainAicpu(index,
                                                 slotToIndex,
                                                 freeSlots,
                                                 freeHead,
                                                 lastQuerySlots,
                                                 reqNum,
                                                 seed,
                                                 indexOut,
                                                 slotToIndexOut,
                                                 freeSlotsOut,
                                                 freeHeadOut,
                                                 uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAsuHbmIndexMaintainAicpu(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAsuHbmIndexMaintainAicpu);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
