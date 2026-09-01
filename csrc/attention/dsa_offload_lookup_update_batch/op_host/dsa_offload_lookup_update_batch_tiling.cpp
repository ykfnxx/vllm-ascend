/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "dsa_offload_lookup_update_batch_tiling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

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
constexpr uint32_t kMappedIndices = 0;
constexpr uint32_t kMissMask = 1;

constexpr uint32_t kReqNumAttr = 0;
constexpr uint32_t kBlockSizeAttr = 1;
constexpr uint32_t kTailBaseAttr = 2;
constexpr uint32_t kFallbackSlotAttr = 3;
constexpr uint32_t kStagingBaseAttr = 4;
constexpr uint32_t kDecodeModeAttr = 5;

constexpr int64_t kIndexCapacity = 128 * 1024;
constexpr int64_t kResidentSlots = 8 * 1024;
constexpr int64_t kSlotCount = 10 * 1024;
constexpr int64_t kFreeSlotCount = 2 * 1024;
constexpr int64_t kTopkWidth = 2 * 1024;
constexpr int64_t kFreeHeadStride = 16;

bool GetShape(
    gert::TilingContext* context,
    bool output,
    uint32_t tensor_index,
    const char* name,
    std::initializer_list<int64_t> expected)
{
    const gert::StorageShape* shape = output
        ? context->GetOutputShape(tensor_index)
        : context->GetInputShape(tensor_index);
    if (shape == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "%s shape is null.", name);
        return false;
    }
    const auto& storage = shape->GetStorageShape();
    if (storage.GetDimNum() != expected.size()) {
        OPS_LOG_E(
            context->GetNodeName(),
            "%s rank is %zu, expected %zu.",
            name,
            storage.GetDimNum(),
            expected.size());
        return false;
    }
    size_t dim = 0;
    for (const int64_t value : expected) {
        if (storage.GetDim(dim) != value) {
            OPS_LOG_E(
                context->GetNodeName(),
                "%s dimension %zu is %ld, expected %ld.",
                name,
                dim,
                storage.GetDim(dim),
                value);
            return false;
        }
        ++dim;
    }
    return true;
}

const int64_t* GetIntAttr(
    gert::TilingContext* context,
    const gert::RuntimeAttrs* attrs,
    uint32_t index,
    const char* name)
{
    const int64_t* value = attrs->GetAttrPointer<int64_t>(index);
    if (value == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "%s is null.", name);
    }
    return value;
}

bool FitsUint32(int64_t value)
{
    return value > 0 &&
           static_cast<uint64_t>(value) <=
               std::numeric_limits<uint32_t>::max();
}

}  // namespace

namespace optiling {

static ge::graphStatus DsaOffloadLookupUpdateBatchTilingFunc(
    gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const int64_t* req_num =
        GetIntAttr(context, attrs, kReqNumAttr, "reqNum");
    const int64_t* block_size =
        GetIntAttr(context, attrs, kBlockSizeAttr, "blockSize");
    const int64_t* tail_base =
        GetIntAttr(context, attrs, kTailBaseAttr, "tailBase");
    const int64_t* fallback_slot =
        GetIntAttr(context, attrs, kFallbackSlotAttr, "fallbackSlot");
    const int64_t* staging_base =
        GetIntAttr(context, attrs, kStagingBaseAttr, "stagingBase");
    const int64_t* decode_mode =
        GetIntAttr(context, attrs, kDecodeModeAttr, "decodeMode");
    if (req_num == nullptr || block_size == nullptr ||
        tail_base == nullptr || fallback_slot == nullptr ||
        staging_base == nullptr || decode_mode == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (!FitsUint32(*req_num) || !FitsUint32(*block_size) ||
        !FitsUint32(*tail_base) || !FitsUint32(*fallback_slot) ||
        !FitsUint32(*staging_base) ||
        (*decode_mode != 0 && *decode_mode != 1)) {
        OPS_LOG_E(context->GetNodeName(), "invalid lookup geometry attrs.");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* index_shape = context->GetInputShape(kIndex);
    const gert::StorageShape* semantic_shape =
        context->GetInputShape(kSemanticTopk);
    if (index_shape == nullptr || semantic_shape == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "required input shape is null.");
        return ge::GRAPH_FAILED;
    }
    const auto& index_storage = index_shape->GetStorageShape();
    const auto& semantic_storage = semantic_shape->GetStorageShape();
    if (index_storage.GetDimNum() != 2 ||
        index_storage.GetDim(1) != kIndexCapacity ||
        semantic_storage.GetDimNum() != 3 ||
        semantic_storage.GetDim(1) != 1 ||
        semantic_storage.GetDim(2) != kTopkWidth) {
        OPS_LOG_E(context->GetNodeName(), "invalid index or semanticTopk shape.");
        return ge::GRAPH_FAILED;
    }
    const int64_t pool_capacity = index_storage.GetDim(0);
    const int64_t query_num = semantic_storage.GetDim(0);
    if (!FitsUint32(pool_capacity) || !FitsUint32(query_num) ||
        *req_num > pool_capacity) {
        OPS_LOG_E(context->GetNodeName(), "invalid pool or query count.");
        return ge::GRAPH_FAILED;
    }
    if (!GetShape(context, false, kSlotToIndex, "slotToIndex",
                  {pool_capacity, kSlotCount}) ||
        !GetShape(context, false, kFreeSlots, "freeSlots",
                  {pool_capacity, kFreeSlotCount}) ||
        !GetShape(context, false, kFreeHead, "freeHead",
                  {pool_capacity, kFreeHeadStride}) ||
        !GetShape(context, false, kRequestRows, "requestRows",
                  {*req_num}) ||
        !GetShape(context, false, kQueryStartLoc, "queryStartLoc",
                  {*req_num + 1}) ||
        !GetShape(context, false, kQueryPositions, "queryPositions",
                  {query_num}) ||
        !GetShape(context, true, kMappedIndices, "mappedIndices",
                  {query_num, 1, kTopkWidth}) ||
        !GetShape(context, true, kMissMask, "missMask",
                  {query_num, 1, kTopkWidth})) {
        return ge::GRAPH_FAILED;
    }

    const int64_t replaceable_base =
        (kResidentSlots + *block_size - 1) / *block_size * *block_size;
    const int64_t expected_tail =
        replaceable_base +
        (kFreeSlotCount + *block_size - 1) /
            *block_size * *block_size;
    if (*tail_base != expected_tail ||
        *fallback_slot != expected_tail + *block_size ||
        *staging_base != *fallback_slot + 1) {
        OPS_LOG_E(context->GetNodeName(), "inconsistent Hot Cache geometry.");
        return ge::GRAPH_FAILED;
    }

    auto* platform_info = context->GetPlatformInfo();
    if (platform_info == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "platform info is null.");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platform_info);
    const uint32_t aiv_count = platform.GetCoreNumAiv();
    if (aiv_count == 0U) {
        OPS_LOG_E(context->GetNodeName(), "No AIV core is available.");
        return ge::GRAPH_FAILED;
    }
    auto* tiling_data =
        context->GetTilingData<DsaOffloadLookupUpdateBatchTilingData>();
    if (tiling_data == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "tiling data is null.");
        return ge::GRAPH_FAILED;
    }
    tiling_data->reqNum = static_cast<uint32_t>(*req_num);
    tiling_data->blockSize = static_cast<uint32_t>(*block_size);
    tiling_data->replaceableBase =
        static_cast<uint32_t>(replaceable_base);
    tiling_data->tailBase = static_cast<uint32_t>(*tail_base);
    tiling_data->fallbackSlot = static_cast<uint32_t>(*fallback_slot);
    tiling_data->stagingBase = static_cast<uint32_t>(*staging_base);
    tiling_data->decodeMode = static_cast<uint32_t>(*decode_mode);

    size_t* workspace_sizes = context->GetWorkspaceSizes(1);
    if (workspace_sizes == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "workspace descriptor is null.");
        return ge::GRAPH_FAILED;
    }
    workspace_sizes[0] = static_cast<size_t>(
        platform.GetLibApiWorkSpaceSize());
    context->SetTilingKey(0);
    context->SetBlockDim(std::min(
        static_cast<uint32_t>(*req_num), aiv_count));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDsaOffloadLookupUpdateBatch(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DsaOffloadLookupUpdateBatch)
    .Tiling(DsaOffloadLookupUpdateBatchTilingFunc)
    .TilingParse<DsaOffloadLookupUpdateBatchCompileInfo>(
        TilingParseForDsaOffloadLookupUpdateBatch);

}  // namespace optiling
