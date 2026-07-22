/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "aclnn_lightning_indexer_decode_update_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerLightningIndexerDecodeUpdatePoolGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *reqPoolEntries, const aclTensor *cacheSlots,
    const aclTensor *actualSeqLengthsKey, const aclTensor *blockTable,
    const aclTensor *topkIndexOut, const aclTensor *topkSlotsOut,
    const aclTensor *missCountOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerLightningIndexerDecodeUpdatePool(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnLightningIndexerDecodeUpdatePoolGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *reqPoolEntries,
    const aclTensor *cacheSlots,
    const aclTensor *actualSeqLengthsKey,
    const aclTensor *blockTable,
    const aclTensor *topkIndexOut,
    const aclTensor *topkSlotsOut,
    const aclTensor *missCountOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerLightningIndexerDecodeUpdatePoolGetWorkspaceSize(
        query, key, weights, reqPoolEntries, cacheSlots, actualSeqLengthsKey, blockTable, topkIndexOut, topkSlotsOut,
        missCountOut, workspaceSize, executor);
}

aclnnStatus aclnnLightningIndexerDecodeUpdatePool(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerLightningIndexerDecodeUpdatePool(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
