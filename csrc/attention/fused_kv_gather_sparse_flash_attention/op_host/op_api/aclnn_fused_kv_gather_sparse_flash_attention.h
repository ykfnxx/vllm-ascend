/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_FUSED_KV_GATHER_SPARSE_FLASH_ATTENTION_H
#define ACLNN_FUSED_KV_GATHER_SPARSE_FLASH_ATTENTION_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnMlaPrologV2WeightNz calculates the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedKvGatherSparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *hostKey,
    const aclTensor *hostKeyRope,
    const aclTensor *hostBlockTable,
    const aclTensor *queryPoolEntries,
    const aclTensor *hotSourceRows,
    const aclTensor *routePlan,
    const aclTensor *installStagingKey,
    const aclTensor *installStagingRope,
    double           scaleValue,
    int64_t          sparseBlockSizeOptional,
    char             *layoutQueryOptional,
    char             *layoutKvOptional,
    int64_t          sparseMode,
    int64_t          preTokens,
    int64_t          nextTokens,
    int64_t          attentionMode,
    bool             returnSoftmaxLse,
    const aclTensor *attentionOut,
    const aclTensor *softmaxMax,
    const aclTensor *softmaxSum,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of ACLNN_FUSED_KV_GATHER_SPARSE_FLASH_ATTENTION_H is used to perform calculations.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedKvGatherSparseFlashAttention(void *workspace,
                                                                            uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            const aclrtStream stream);


#ifdef __cplusplus
}
#endif

#endif // ACLNN_FUSED_KV_GATHER_SPARSE_FLASH_ATTENTION_H
