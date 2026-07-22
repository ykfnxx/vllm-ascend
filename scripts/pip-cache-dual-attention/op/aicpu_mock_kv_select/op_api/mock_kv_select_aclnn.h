#ifndef MOCK_KV_SELECT_ACLNN_H
#define MOCK_KV_SELECT_ACLNN_H

#include <cstdint>

#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnMockKVSelectGetWorkspaceSize(
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
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnMockKVSelect(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // MOCK_KV_SELECT_ACLNN_H
