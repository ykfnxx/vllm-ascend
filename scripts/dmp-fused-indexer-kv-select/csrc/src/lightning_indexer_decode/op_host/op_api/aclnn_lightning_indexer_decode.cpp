/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "aclnn_lightning_indexer_decode.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerLightningIndexerDecodeGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *actualSeqLengthsKey, const aclTensor *blockTable,
    const aclTensor *sparseIndicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerLightningIndexerDecode(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnLightningIndexerDecodeGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *actualSeqLengthsKey,
    const aclTensor *blockTable,
    const aclTensor *sparseIndicesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerLightningIndexerDecodeGetWorkspaceSize(
        query, key, weights, actualSeqLengthsKey, blockTable, sparseIndicesOut, workspaceSize, executor);
}

aclnnStatus aclnnLightningIndexerDecode(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerLightningIndexerDecode(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
