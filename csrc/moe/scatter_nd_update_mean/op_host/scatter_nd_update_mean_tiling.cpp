/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "scatter_nd_update_mean_tiling.h"

#include "error/ops_error.h"
#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

namespace optiling {
namespace {
constexpr uint32_t INPUT_FLAT_KEY_CACHE_INDEX = 0;
constexpr uint32_t INPUT_INDICES_INDEX = 1;
constexpr uint32_t INPUT_UPDATES_INDEX = 2;
constexpr uint32_t OUTPUT_KEY_MEAN_INDEX = 0;
constexpr uint32_t ATTR_BLOCK_SIZE_INDEX = 0;
constexpr uint32_t ATTR_UPDATE_CACHE_IN_KERNEL_INDEX = 1;
} // namespace

static ge::graphStatus ScatterNdUpdateMeanTilingFunc(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("ScatterNdUpdateMean", "Tiling context is null."),
               return ge::GRAPH_FAILED);

    const gert::StorageShape *flatShape = context->GetInputShape(INPUT_FLAT_KEY_CACHE_INDEX);
    const gert::StorageShape *indicesShape = context->GetInputShape(INPUT_INDICES_INDEX);
    const gert::StorageShape *updatesShape = context->GetInputShape(INPUT_UPDATES_INDEX);
    const gert::StorageShape *keyMeanShape = context->GetOutputShape(OUTPUT_KEY_MEAN_INDEX);
    OPS_LOG_E_IF_NULL(context, flatShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, indicesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, updatesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, keyMeanShape, return ge::GRAPH_FAILED);

    const gert::Shape &flatStorage = flatShape->GetStorageShape();
    const gert::Shape &indicesStorage = indicesShape->GetStorageShape();
    const gert::Shape &updatesStorage = updatesShape->GetStorageShape();
    const gert::Shape &keyMeanStorage = keyMeanShape->GetStorageShape();
    OPS_ERR_IF(flatStorage.GetDimNum() != 2,
               OPS_LOG_E(context->GetNodeName(), "flat_key_cache must be 2-D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(indicesStorage.GetDimNum() != 2 || indicesStorage.GetDim(1) != 1,
               OPS_LOG_E(context->GetNodeName(), "indices must be [num_updates, 1]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(updatesStorage.GetDimNum() != 2,
               OPS_LOG_E(context->GetNodeName(), "updates must be 2-D: [num_updates, head_dim]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyMeanStorage.GetDimNum() != 4 || keyMeanStorage.GetDim(1) != 1,
               OPS_LOG_E(context->GetNodeName(), "key_mean must be [num_blocks, 1, kv_heads, head_dim]."),
               return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t *blockSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_INDEX);
    const bool *updateCacheInKernelPtr = attrs->GetAttrPointer<bool>(ATTR_UPDATE_CACHE_IN_KERNEL_INDEX);
    OPS_LOG_E_IF_NULL(context, blockSizePtr, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, updateCacheInKernelPtr, return ge::GRAPH_FAILED);
    OPS_ERR_IF(*blockSizePtr <= 0,
               OPS_LOG_E(context->GetNodeName(), "block_size must be positive."),
               return ge::GRAPH_FAILED);

    const uint32_t numUpdates = static_cast<uint32_t>(indicesStorage.GetDim(0));
    const uint32_t headDim = static_cast<uint32_t>(flatStorage.GetDim(1));
    const uint32_t blockSize = static_cast<uint32_t>(*blockSizePtr);
    const uint32_t kvHeads = static_cast<uint32_t>(keyMeanStorage.GetDim(2));
    const uint32_t numBlocks = static_cast<uint32_t>(keyMeanStorage.GetDim(0));

    OPS_ERR_IF(updatesStorage.GetDim(0) != indicesStorage.GetDim(0) ||
                   updatesStorage.GetDim(1) != flatStorage.GetDim(1),
               OPS_LOG_E(context->GetNodeName(), "updates shape must be [num_updates, head_dim]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyMeanStorage.GetDim(3) != flatStorage.GetDim(1),
               OPS_LOG_E(context->GetNodeName(), "key_mean head_dim must match flat_key_cache head_dim."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(flatStorage.GetDim(0) < static_cast<int64_t>(numBlocks) * blockSize * kvHeads,
               OPS_LOG_E(context->GetNodeName(), "flat_key_cache rows are smaller than numBlocks*blockSize*kvHeads."),
               return ge::GRAPH_FAILED);

    ScatterNdUpdateMeanTilingData *tiling = context->GetTilingData<ScatterNdUpdateMeanTilingData>();
    OPS_LOG_E_IF_NULL(context, tiling, return ge::GRAPH_FAILED);
    tiling->numUpdates = numUpdates;
    tiling->headDim = headDim;
    tiling->blockSize = blockSize;
    tiling->kvHeads = kvHeads;
    tiling->numBlocks = numBlocks;
    tiling->updateCacheInKernel = *updateCacheInKernelPtr ? 1U : 0U;
    tiling->invBlockSize = 1.0f / static_cast<float>(blockSize);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t usedCoreNum = std::min(std::max(numUpdates, 1U), std::max(aivNum, 1U));
    context->SetBlockDim(usedCoreNum);
    context->SetScheduleMode(1);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForScatterNdUpdateMean(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct ScatterNdUpdateMeanCompileInfo {};

IMPL_OP_OPTILING(ScatterNdUpdateMean)
    .Tiling(ScatterNdUpdateMeanTilingFunc)
    .TilingParse<ScatterNdUpdateMeanCompileInfo>(TilingPrepareForScatterNdUpdateMean);

} // namespace optiling
