/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "dsa_sparse_lookup_update_tiling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t kIndex = 0;
constexpr uint32_t kSlotToIndex = 1;
constexpr uint32_t kFreeSlots = 2;
constexpr uint32_t kFreeHead = 3;
constexpr uint32_t kReqPoolEntries = 4;
constexpr uint32_t kQueryIndex = 5;
constexpr uint32_t kLookupMask = 6;
constexpr uint32_t kSlotOut = 0;
constexpr uint32_t kMissOut = 1;
constexpr uint32_t kReqNumAttr = 0;

constexpr int64_t kIndexCapacity = 128 * 1024;
constexpr int64_t kSlotCount = 10 * 1024;
constexpr int64_t kFreeSlotCount = 2 * 1024;
constexpr int64_t kQueryCount = 2 * 1024;
constexpr int64_t kFreeHeadStride = 16;

bool GetInputOneDim(
    gert::TilingContext* context,
    uint32_t input_index,
    int64_t& dim0)
{
    const gert::StorageShape* shape =
        context->GetInputShape(input_index);
    if (shape == nullptr) {
        return false;
    }
    const auto& storage_shape = shape->GetStorageShape();
    if (storage_shape.GetDimNum() != 1) {
        return false;
    }
    dim0 = storage_shape.GetDim(0);
    return dim0 > 0;
}

bool GetInputTwoDims(
    gert::TilingContext* context,
    uint32_t input_index,
    int64_t& dim0,
    int64_t& dim1)
{
    const gert::StorageShape* shape =
        context->GetInputShape(input_index);
    if (shape == nullptr) {
        return false;
    }
    const auto& storage_shape = shape->GetStorageShape();
    if (storage_shape.GetDimNum() != 2) {
        return false;
    }
    dim0 = storage_shape.GetDim(0);
    dim1 = storage_shape.GetDim(1);
    return dim0 > 0 && dim1 > 0;
}

bool GetOutputTwoDims(
    gert::TilingContext* context,
    uint32_t output_index,
    int64_t& dim0,
    int64_t& dim1)
{
    const gert::StorageShape* shape =
        context->GetOutputShape(output_index);
    if (shape == nullptr) {
        return false;
    }
    const auto& storage_shape = shape->GetStorageShape();
    if (storage_shape.GetDimNum() != 2) {
        return false;
    }
    dim0 = storage_shape.GetDim(0);
    dim1 = storage_shape.GetDim(1);
    return dim0 > 0 && dim1 > 0;
}

bool RequireInputShape(
    gert::TilingContext* context,
    uint32_t input_index,
    int64_t expected0,
    int64_t expected1)
{
    int64_t actual0 = 0;
    int64_t actual1 = 0;
    if (!GetInputTwoDims(
            context, input_index, actual0, actual1)) {
        return false;
    }
    if (actual0 != expected0 || actual1 != expected1) {
        return false;
    }
    return true;
}

bool RequireOutputShape(
    gert::TilingContext* context,
    uint32_t output_index,
    int64_t expected0,
    int64_t expected1)
{
    int64_t actual0 = 0;
    int64_t actual1 = 0;
    if (!GetOutputTwoDims(
            context, output_index, actual0, actual1)) {
        return false;
    }
    if (actual0 != expected0 || actual1 != expected1) {
        return false;
    }
    return true;
}

}  // namespace

namespace optiling {

static ge::graphStatus DsaSparseLookupUpdateTilingFunc(
    gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* req_num_attr =
        attrs->GetAttrPointer<int64_t>(kReqNumAttr);
    if (req_num_attr == nullptr || *req_num_attr <= 0 ||
        static_cast<uint64_t>(*req_num_attr) >
            std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const int64_t req_num = *req_num_attr;

    int64_t pool_capacity = 0;
    int64_t index_width = 0;
    if (!GetInputTwoDims(
            context, kIndex, pool_capacity, index_width)) {
        return ge::GRAPH_FAILED;
    }
    if (index_width != kIndexCapacity) {
        return ge::GRAPH_FAILED;
    }
    if (req_num > pool_capacity) {
        return ge::GRAPH_FAILED;
    }
    if (pool_capacity > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    int64_t req_entries = 0;
    if (!GetInputOneDim(
            context, kReqPoolEntries, req_entries) ||
        req_entries != req_num ||
        !RequireInputShape(
            context, kSlotToIndex, pool_capacity, kSlotCount) ||
        !RequireInputShape(
            context, kFreeSlots, pool_capacity, kFreeSlotCount) ||
        !RequireInputShape(
            context, kFreeHead, pool_capacity, kFreeHeadStride) ||
        !RequireInputShape(
            context, kQueryIndex, req_num, kQueryCount) ||
        !RequireInputShape(
            context, kLookupMask, req_num, kQueryCount) ||
        !RequireOutputShape(
            context, kSlotOut, req_num, kQueryCount) ||
        !RequireOutputShape(
            context, kMissOut, req_num, kQueryCount)) {
        return ge::GRAPH_FAILED;
    }

    auto* platform_info = context->GetPlatformInfo();
    if (platform_info == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform =
        platform_ascendc::PlatformAscendC(platform_info);
    const uint32_t aiv_count = platform.GetCoreNumAiv();
    if (aiv_count == 0U) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling_data =
        context->GetTilingData<DsaSparseLookupUpdateTilingData>();
    if (tiling_data == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling_data->reqNum = static_cast<uint32_t>(req_num);
    tiling_data->poolCapacity =
        static_cast<uint32_t>(pool_capacity);

    const uint64_t workspace_bytes =
        static_cast<uint64_t>(
            platform.GetLibApiWorkSpaceSize());
    if (workspace_bytes >
        static_cast<uint64_t>(
            std::numeric_limits<size_t>::max())) {
        return ge::GRAPH_FAILED;
    }
    size_t* system_workspace = context->GetWorkspaceSizes(1);
    if (system_workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    system_workspace[0] = static_cast<size_t>(workspace_bytes);

    context->SetTilingKey(0);
    context->SetBlockDim(std::min(
        static_cast<uint32_t>(req_num), aiv_count));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDsaSparseLookupUpdate(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DsaSparseLookupUpdate)
    .Tiling(DsaSparseLookupUpdateTilingFunc)
    .TilingParse<DsaSparseLookupUpdateCompileInfo>(
        TilingParseForDsaSparseLookupUpdate);

}  // namespace optiling
