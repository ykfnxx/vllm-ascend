#include "mock_kv_select_aclnn.h"

#include <initializer_list>

#include "acl/acl.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

namespace {
OP_TYPE_REGISTER(MockKVSelect);
constexpr const char *CUST_AICPU_LIB_ID = "asn_aicpu_kernels";

bool AllTensorNotNull(std::initializer_list<const aclTensor *> tensors)
{
    for (const aclTensor *tensor : tensors) {
        if (tensor == nullptr) {
            return false;
        }
    }
    return true;
}
}  // namespace

extern "C" aclnnStatus aclnnMockKVSelectGetWorkspaceSize(
    const aclTensor *selectionKRope,
    const aclTensor *selectionKvCache,
    const aclTensor *selectionKvBlockTable,
    const aclTensor *selectionKvBlockStatus,
    const aclTensor *selectionTopkIndices,
    const aclTensor *fullKRope,
    const aclTensor *fullKvCache,
    const aclTensor *fullKvBlockTable,
    const aclTensor *fullKvActualSeq,
    const aclTensor *fullQActualSeq,
    int64_t selectionTopkBlockSize,
    int64_t mockWaitUs,
    aclTensor *hitSparseIndices,
    aclTensor *missTopkIndices,
    aclTensor *missInsertIndices,
    aclTensor *hitActualSeq,
    aclTensor *missActualSeq,
    aclTensor *missCount,
    aclTensor *hitCount,
    aclTensor *selectionStatusEmpty,
    uint64_t *workspaceSize,
    aclOpExecutor **executorOut)
{
    if (workspaceSize == nullptr || executorOut == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (!AllTensorNotNull({selectionKRope, selectionKvCache, selectionKvBlockTable,
                           selectionKvBlockStatus, selectionTopkIndices, fullKRope,
                           fullKvCache, fullKvBlockTable, fullKvActualSeq, fullQActualSeq,
                           hitSparseIndices, missTopkIndices, missInsertIndices, hitActualSeq,
                           missActualSeq, missCount, hitCount, selectionStatusEmpty})) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }

    aclOpExecutor *executor = uniqueExecutor.get();
    static internal::AicpuTaskSpace space("MockKVSelect");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        MockKVSelect,
        OP_ATTR_NAMES({"selection_topk_block_size", "mock_wait_us", "cust_aicpu"}),
        OP_INPUT(selectionKRope, selectionKvCache, selectionKvBlockTable, selectionKvBlockStatus,
                 selectionTopkIndices, fullKRope, fullKvCache, fullKvBlockTable,
                 fullKvActualSeq, fullQActualSeq),
        OP_OUTPUT(hitSparseIndices, missTopkIndices, missInsertIndices, hitActualSeq,
                  missActualSeq, missCount, hitCount, selectionStatusEmpty),
        OP_ATTR(selectionTopkBlockSize, mockWaitUs, CUST_AICPU_LIB_ID));
    if (ret != OK) {
        return ret;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executorOut);
    return OK;
}

extern "C" aclnnStatus aclnnMockKVSelect(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
