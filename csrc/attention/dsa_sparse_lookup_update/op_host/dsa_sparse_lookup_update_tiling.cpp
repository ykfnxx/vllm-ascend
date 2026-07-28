/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "dsa_sparse_lookup_update_tiling.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t kTokenToHot = 0;
constexpr uint32_t kHotToToken = 1;
constexpr uint32_t kLruSlots = 2;
constexpr uint32_t kStateSeatEpoch = 3;
constexpr uint32_t kRowToCacheSeat = 4;
constexpr uint32_t kRowSeatEpoch = 5;
constexpr uint32_t kQueryPositions = 6;
constexpr uint32_t kQueryToRow = 7;
constexpr uint32_t kQueryToLane = 8;
constexpr uint32_t kQueryValidMask = 9;
constexpr uint32_t kValidTopkCounts = 10;
constexpr uint32_t kSeqLens = 11;
constexpr uint32_t kTopkPositions = 12;
constexpr uint32_t kResolvedHotIndices = 13;
constexpr uint32_t kMissMask = 14;
constexpr uint32_t kOpWorkspace = 15;

constexpr uint32_t kSimtThreads = 256;
constexpr uint32_t kMaxQueryLanes = 4;

bool GetOneDim(
    gert::TilingContext* context,
    uint32_t input_index,
    const char* input_name,
    int64_t& dim0)
{
    const gert::StorageShape* shape = context->GetInputShape(input_index);
    if (shape == nullptr) {
        OPS_LOG_E(
            context->GetNodeName(), "%s shape is null.", input_name);
        return false;
    }
    const auto& storage_shape = shape->GetStorageShape();
    if (storage_shape.GetDimNum() != 1) {
        OPS_LOG_E(
            context->GetNodeName(), "%s must be rank 1.", input_name);
        return false;
    }
    dim0 = storage_shape.GetDim(0);
    return dim0 > 0;
}

bool GetTwoDims(
    gert::TilingContext* context,
    uint32_t input_index,
    const char* input_name,
    int64_t& dim0,
    int64_t& dim1)
{
    const gert::StorageShape* shape = context->GetInputShape(input_index);
    if (shape == nullptr) {
        OPS_LOG_E(
            context->GetNodeName(), "%s shape is null.", input_name);
        return false;
    }
    const auto& storage_shape = shape->GetStorageShape();
    if (storage_shape.GetDimNum() != 2) {
        OPS_LOG_E(
            context->GetNodeName(), "%s must be rank 2.", input_name);
        return false;
    }
    dim0 = storage_shape.GetDim(0);
    dim1 = storage_shape.GetDim(1);
    return dim0 > 0 && dim1 > 0;
}

bool SameOneDim(
    gert::TilingContext* context,
    uint32_t input_index,
    const char* input_name,
    int64_t expected)
{
    int64_t actual = 0;
    if (!GetOneDim(context, input_index, input_name, actual)) {
        return false;
    }
    if (actual != expected) {
        OPS_LOG_E(
            context->GetNodeName(),
            "%s has incompatible length %ld; expected %ld.",
            input_name,
            actual,
            expected);
        return false;
    }
    return true;
}

bool SameTwoDims(
    gert::TilingContext* context,
    uint32_t input_index,
    const char* input_name,
    int64_t expected0,
    int64_t expected1)
{
    int64_t actual0 = 0;
    int64_t actual1 = 0;
    if (!GetTwoDims(
            context, input_index, input_name, actual0, actual1)) {
        return false;
    }
    if (actual0 != expected0 || actual1 != expected1) {
        OPS_LOG_E(
            context->GetNodeName(),
            "%s has incompatible shape [%ld, %ld]; expected [%ld, %ld].",
            input_name,
            actual0,
            actual1,
            expected0,
            expected1);
        return false;
    }
    return true;
}

}  // namespace

namespace optiling {

static ge::graphStatus DsaSparseLookupUpdateTilingFunc(
    gert::TilingContext* context)
{
    int64_t seat_capacity = 0;
    int64_t token_position_capacity = 0;
    int64_t hot_seat_capacity = 0;
    int64_t evictable_slot_count = 0;
    int64_t query_capacity = 0;
    int64_t request_capacity = 0;
    int64_t topk_query_capacity = 0;
    int64_t topk_count = 0;
    int64_t workspace_request_capacity = 0;
    int64_t workspace_stride = 0;

    if (!GetTwoDims(
            context,
            kTokenToHot,
            "tokenToHot",
            seat_capacity,
            token_position_capacity) ||
        !GetTwoDims(
            context,
            kHotToToken,
            "hotToToken",
            hot_seat_capacity,
            evictable_slot_count) ||
        !GetOneDim(
            context,
            kQueryPositions,
            "queryPositions",
            query_capacity) ||
        !GetOneDim(
            context,
            kRowToCacheSeat,
            "rowToCacheSeat",
            request_capacity) ||
        !GetTwoDims(
            context,
            kTopkPositions,
            "topkPositions",
            topk_query_capacity,
            topk_count) ||
        !GetTwoDims(
            context,
            kOpWorkspace,
            "workspace",
            workspace_request_capacity,
            workspace_stride)) {
        return ge::GRAPH_FAILED;
    }

    if (hot_seat_capacity != seat_capacity) {
        OPS_LOG_E(
            context->GetNodeName(),
            "tokenToHot and hotToToken seat capacities differ.");
        return ge::GRAPH_FAILED;
    }
    if (topk_query_capacity != query_capacity) {
        OPS_LOG_E(
            context->GetNodeName(),
            "topkPositions query dimension differs from queryPositions.");
        return ge::GRAPH_FAILED;
    }
    if (query_capacity % request_capacity != 0) {
        OPS_LOG_E(
            context->GetNodeName(),
            "query capacity %ld is not divisible by request capacity %ld.",
            query_capacity,
            request_capacity);
        return ge::GRAPH_FAILED;
    }

    const int64_t query_lane_capacity =
        query_capacity / request_capacity;
    if (query_lane_capacity <= 0 ||
        query_lane_capacity > static_cast<int64_t>(kMaxQueryLanes)) {
        OPS_LOG_E(
            context->GetNodeName(),
            "query lane capacity must be in [1, %u], got %ld.",
            kMaxQueryLanes,
            query_lane_capacity);
        return ge::GRAPH_FAILED;
    }

    const uint64_t protected_union_width =
        static_cast<uint64_t>(query_lane_capacity) *
        static_cast<uint64_t>(topk_count);
    if (protected_union_width >
        static_cast<uint64_t>(evictable_slot_count)) {
        OPS_LOG_E(
            context->GetNodeName(),
            "evictable slot count %ld is smaller than T*K=%lu.",
            evictable_slot_count,
            protected_union_width);
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint64_t>(query_capacity) *
            static_cast<uint64_t>(topk_count) >
        static_cast<uint64_t>(
            std::numeric_limits<int32_t>::max() - 2)) {
        OPS_LOG_E(
            context->GetNodeName(),
            "Q*K does not fit the deterministic claim encoding.");
        return ge::GRAPH_FAILED;
    }

    const uint64_t required_workspace_stride =
        3ULL * static_cast<uint64_t>(evictable_slot_count) +
        3ULL * kSimtThreads + 4ULL;
    if (workspace_request_capacity != request_capacity ||
        static_cast<uint64_t>(workspace_stride) !=
            required_workspace_stride) {
        OPS_LOG_E(
            context->GetNodeName(),
            "workspace must have shape [R, 3*S+3*256+4].");
        return ge::GRAPH_FAILED;
    }
    const uint64_t uint32_max =
        std::numeric_limits<uint32_t>::max();
    if (static_cast<uint64_t>(seat_capacity) > uint32_max ||
        static_cast<uint64_t>(token_position_capacity) >
            uint32_max ||
        static_cast<uint64_t>(evictable_slot_count) >
            uint32_max ||
        static_cast<uint64_t>(query_capacity) > uint32_max ||
        static_cast<uint64_t>(request_capacity) > uint32_max ||
        static_cast<uint64_t>(topk_count) > uint32_max ||
        static_cast<uint64_t>(workspace_stride) > uint32_max) {
        OPS_LOG_E(
            context->GetNodeName(),
            "one or more dimensions exceed the uint32 tiling ABI.");
        return ge::GRAPH_FAILED;
    }

    if (!SameTwoDims(
            context,
            kLruSlots,
            "lruSlots",
            seat_capacity,
            evictable_slot_count) ||
        !SameOneDim(
            context,
            kStateSeatEpoch,
            "stateSeatEpoch",
            seat_capacity) ||
        !SameOneDim(
            context,
            kRowSeatEpoch,
            "rowSeatEpoch",
            request_capacity) ||
        !SameOneDim(
            context,
            kSeqLens,
            "seqLens",
            request_capacity) ||
        !SameOneDim(
            context,
            kQueryToRow,
            "queryToRow",
            query_capacity) ||
        !SameOneDim(
            context,
            kQueryToLane,
            "queryToLane",
            query_capacity) ||
        !SameOneDim(
            context,
            kQueryValidMask,
            "queryValidMask",
            query_capacity) ||
        !SameOneDim(
            context,
            kValidTopkCounts,
            "validTopkCounts",
            query_capacity) ||
        !SameTwoDims(
            context,
            kResolvedHotIndices,
            "resolvedHotIndices",
            query_capacity,
            topk_count) ||
        !SameTwoDims(
            context,
            kMissMask,
            "missMask",
            query_capacity,
            topk_count)) {
        return ge::GRAPH_FAILED;
    }

    auto* platform_info = context->GetPlatformInfo();
    if (platform_info == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "platform info is null.");
        return ge::GRAPH_FAILED;
    }
    auto platform =
        platform_ascendc::PlatformAscendC(platform_info);
    const uint32_t aiv_count = platform.GetCoreNumAiv();
    if (aiv_count == 0U) {
        OPS_LOG_E(
            context->GetNodeName(),
            "No AIV core is available for dsa_sparse_lookup_update.");
        return ge::GRAPH_FAILED;
    }

    auto* tiling_data =
        context->GetTilingData<DsaSparseLookupUpdateTilingData>();
    if (tiling_data == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "tiling data is null.");
        return ge::GRAPH_FAILED;
    }
    tiling_data->seatCapacity =
        static_cast<uint32_t>(seat_capacity);
    tiling_data->tokenPositionCapacity =
        static_cast<uint32_t>(token_position_capacity);
    tiling_data->evictableSlotCount =
        static_cast<uint32_t>(evictable_slot_count);
    tiling_data->queryCapacity =
        static_cast<uint32_t>(query_capacity);
    tiling_data->requestCapacity =
        static_cast<uint32_t>(request_capacity);
    tiling_data->queryLaneCapacity =
        static_cast<uint32_t>(query_lane_capacity);
    tiling_data->topkCount =
        static_cast<uint32_t>(topk_count);
    tiling_data->workspaceStride =
        static_cast<uint32_t>(workspace_stride);

    size_t* system_workspace = context->GetWorkspaceSizes(1);
    if (system_workspace != nullptr) {
        system_workspace[0] = 0;
    }
    // The kernel has no template-specialized variants, so launch its
    // default function entry.
    context->SetTilingKey(0);
    const uint32_t request_count =
        static_cast<uint32_t>(request_capacity);
    context->SetBlockDim(
        request_count < aiv_count ? request_count : aiv_count);
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
