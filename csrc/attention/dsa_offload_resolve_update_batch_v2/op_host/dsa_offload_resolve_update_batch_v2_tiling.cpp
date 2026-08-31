/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "dsa_offload_resolve_update_batch_v2_tiling.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kIndex = 0;
constexpr uint32_t kSlotToIndex = 1;
constexpr uint32_t kFreeSlots = 2;
constexpr uint32_t kFreeHead = 3;
constexpr uint32_t kRequestRows = 4;
constexpr uint32_t kQueryStartLoc = 5;
constexpr uint32_t kQueryPositions = 6;
constexpr uint32_t kSemanticTopk = 7;
constexpr int64_t kIndexCapacity = 128 * 1024;
constexpr int64_t kSlotCount = 10 * 1024;
constexpr int64_t kFreeSlotCount = 2 * 1024;
constexpr int64_t kQueryWidth = 2 * 1024;
constexpr int64_t kFreeHeadStride = 16;
constexpr int64_t kBlockSize = 128;

bool ShapeEquals(
    gert::TilingContext* context,
    uint32_t index,
    std::initializer_list<int64_t> expected)
{
    const auto* storage = context->GetInputShape(index);
    if (storage == nullptr) {
        return false;
    }
    const auto& shape = storage->GetStorageShape();
    if (shape.GetDimNum() != expected.size()) {
        return false;
    }
    size_t dim = 0;
    for (int64_t value : expected) {
        if (shape.GetDim(dim++) != value) {
            return false;
        }
    }
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus DsaOffloadResolveUpdateBatchV2Tiling(
    gert::TilingContext* context)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* req_num = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* block_size = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* decode_mode = attrs->GetAttrPointer<int64_t>(2);
    if (req_num == nullptr || block_size == nullptr || decode_mode == nullptr ||
        *req_num <= 0 || *block_size != kBlockSize ||
        (*decode_mode != 0 && *decode_mode != 1)) {
        return ge::GRAPH_FAILED;
    }
    const auto* index_shape = context->GetInputShape(kIndex);
    const auto* topk_shape = context->GetInputShape(kSemanticTopk);
    if (index_shape == nullptr || topk_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& index = index_shape->GetStorageShape();
    const auto& topk = topk_shape->GetStorageShape();
    if (index.GetDimNum() != 2 || index.GetDim(1) != kIndexCapacity ||
        topk.GetDimNum() != 3 || topk.GetDim(1) != 1 ||
        topk.GetDim(2) != kQueryWidth) {
        return ge::GRAPH_FAILED;
    }
    const int64_t pool_capacity = index.GetDim(0);
    const int64_t query_num = topk.GetDim(0);
    if (*req_num > pool_capacity || query_num <= 0 ||
        !ShapeEquals(context, kSlotToIndex, {pool_capacity, kSlotCount}) ||
        !ShapeEquals(context, kFreeSlots, {pool_capacity, kFreeSlotCount}) ||
        !ShapeEquals(context, kFreeHead, {pool_capacity, kFreeHeadStride}) ||
        !ShapeEquals(context, kRequestRows, {*req_num}) ||
        !ShapeEquals(context, kQueryStartLoc, {*req_num + 1}) ||
        !ShapeEquals(context, kQueryPositions, {query_num})) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    const uint32_t aiv_count = platform.GetCoreNumAiv();
    if (aiv_count == 0) {
        return ge::GRAPH_FAILED;
    }
    auto* data =
        context->GetTilingData<DsaOffloadResolveUpdateBatchV2TilingData>();
    data->reqNum = static_cast<uint32_t>(*req_num);
    data->poolCapacity = static_cast<uint32_t>(pool_capacity);
    data->queryNum = static_cast<uint32_t>(query_num);
    data->blockSize = static_cast<uint32_t>(*block_size);
    data->decodeMode = static_cast<uint32_t>(*decode_mode);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    context->SetTilingKey(0);
    context->SetBlockDim(std::min(data->reqNum, aiv_count));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseDsaOffloadResolveUpdateBatchV2(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DsaOffloadResolveUpdateBatchV2)
    .Tiling(DsaOffloadResolveUpdateBatchV2Tiling)
    .TilingParse<DsaOffloadResolveUpdateBatchV2CompileInfo>(
        ParseDsaOffloadResolveUpdateBatchV2);
}  // namespace optiling
