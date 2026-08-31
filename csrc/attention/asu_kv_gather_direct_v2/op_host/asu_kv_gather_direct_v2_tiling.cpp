/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
 */

#include "asu_kv_gather_direct_v2_tiling.h"

#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kDestinationKv = 0;
constexpr uint32_t kDestinationRope = 1;
constexpr uint32_t kHotTable = 2;
constexpr uint32_t kSourceKv = 3;
constexpr uint32_t kSourceRope = 4;
constexpr uint32_t kSourceTable = 5;
constexpr uint32_t kRequestRows = 6;
constexpr uint32_t kQueryStartLoc = 7;
constexpr uint32_t kSemanticTopk = 8;
constexpr uint32_t kMappedIndices = 9;
constexpr uint32_t kGatherMask = 10;
constexpr uint32_t kJitterSeed = 0x9E3779B9U;

const gert::Shape* GetShape(gert::TilingContext* context, uint32_t index)
{
    const auto* storage = context->GetInputShape(index);
    return storage == nullptr ? nullptr : &storage->GetStorageShape();
}

uint32_t TypeSize(ge::DataType type)
{
    return type == ge::DT_INT8 ? 1U : 2U;
}
}  // namespace

namespace optiling {
static ge::graphStatus AsuKvGatherDirectV2Tiling(
    gert::TilingContext* context)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* block_size = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* req_num = attrs->GetAttrPointer<int64_t>(1);
    if (block_size == nullptr || req_num == nullptr ||
        *block_size != 128 || *req_num <= 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* destination_kv = GetShape(context, kDestinationKv);
    const gert::Shape* destination_rope =
        GetShape(context, kDestinationRope);
    const gert::Shape* hot_table = GetShape(context, kHotTable);
    const gert::Shape* source_kv = GetShape(context, kSourceKv);
    const gert::Shape* source_rope = GetShape(context, kSourceRope);
    const gert::Shape* source_table = GetShape(context, kSourceTable);
    const gert::Shape* request_rows = GetShape(context, kRequestRows);
    const gert::Shape* query_start = GetShape(context, kQueryStartLoc);
    const gert::Shape* topk = GetShape(context, kSemanticTopk);
    const gert::Shape* mapped = GetShape(context, kMappedIndices);
    const gert::Shape* mask = GetShape(context, kGatherMask);
    if (destination_kv == nullptr || destination_rope == nullptr ||
        hot_table == nullptr || source_kv == nullptr ||
        source_rope == nullptr || source_table == nullptr ||
        request_rows == nullptr || query_start == nullptr ||
        topk == nullptr || mapped == nullptr || mask == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (destination_kv->GetDimNum() != 3 ||
        destination_rope->GetDimNum() != 3 ||
        source_kv->GetDimNum() != 3 || source_rope->GetDimNum() != 3 ||
        hot_table->GetDimNum() != 2 || source_table->GetDimNum() != 2 ||
        request_rows->GetDimNum() != 1 ||
        query_start->GetDimNum() != 1 || topk->GetDimNum() != 3 ||
        mapped->GetDimNum() != 3 || mask->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    const int64_t query_num = topk->GetDim(0);
    const int64_t pool_capacity = hot_table->GetDim(0);
    if (query_num <= 0 || pool_capacity < *req_num ||
        hot_table->GetDim(1) <= 0 || source_table->GetDim(1) <= 0 ||
        destination_kv->GetDim(0) <= 0 || source_kv->GetDim(0) <= 0 ||
        destination_kv->GetDim(2) <= 0 || destination_rope->GetDim(2) <= 0 ||
        topk->GetDim(1) != 1 ||
        topk->GetDim(2) != 2048 || mapped->GetDim(0) != query_num ||
        mapped->GetDim(1) != 1 || mapped->GetDim(2) != 2048 ||
        mask->GetDim(0) != query_num || mask->GetDim(1) != 1 ||
        mask->GetDim(2) != 2048 ||
        request_rows->GetDim(0) != *req_num ||
        query_start->GetDim(0) != *req_num + 1 ||
        source_table->GetDim(0) != pool_capacity ||
        destination_kv->GetDim(1) != *block_size ||
        destination_rope->GetDim(1) != *block_size ||
        source_kv->GetDim(1) != *block_size ||
        source_rope->GetDim(1) != *block_size ||
        destination_kv->GetDim(0) != destination_rope->GetDim(0) ||
        source_kv->GetDim(0) != source_rope->GetDim(0) ||
        destination_kv->GetDim(2) != source_kv->GetDim(2) ||
        destination_rope->GetDim(2) != source_rope->GetDim(2)) {
        return ge::GRAPH_FAILED;
    }

    const auto* destination_kv_desc = context->GetInputDesc(kDestinationKv);
    const auto* destination_rope_desc =
        context->GetInputDesc(kDestinationRope);
    const auto* source_kv_desc = context->GetInputDesc(kSourceKv);
    const auto* source_rope_desc = context->GetInputDesc(kSourceRope);
    if (destination_kv_desc == nullptr || destination_rope_desc == nullptr ||
        source_kv_desc == nullptr || source_rope_desc == nullptr ||
        source_kv_desc->GetDataType() !=
            destination_kv_desc->GetDataType() ||
        source_rope_desc->GetDataType() !=
            destination_rope_desc->GetDataType()) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType kv_type = destination_kv_desc->GetDataType();
    const ge::DataType rope_type = destination_rope_desc->GetDataType();
    const bool supported =
        (kv_type == ge::DT_FLOAT16 && rope_type == ge::DT_FLOAT16) ||
        (kv_type == ge::DT_BF16 && rope_type == ge::DT_BF16) ||
        (kv_type == ge::DT_INT8 && rope_type == ge::DT_BF16);
    const uint64_t kv_bytes =
        destination_kv->GetDim(2) * TypeSize(kv_type);
    const uint64_t rope_bytes =
        destination_rope->GetDim(2) * TypeSize(rope_type);
    if (!supported || kv_bytes % 32U != 0U || rope_bytes % 32U != 0U) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    const uint32_t aiv_count = platform.GetCoreNumAiv();
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    if (aiv_count == 0 || kv_bytes + rope_bytes > ub_size) {
        return ge::GRAPH_FAILED;
    }

    AsuKvGatherDirectV2TilingData data;
    data.set_reqNum(static_cast<uint32_t>(*req_num));
    data.set_queryNum(static_cast<uint32_t>(query_num));
    data.set_topkWidth(2048U);
    data.set_blockSize(static_cast<uint32_t>(*block_size));
    data.set_sourceTableWidth(
        static_cast<uint32_t>(source_table->GetDim(1)));
    data.set_destinationTableWidth(
        static_cast<uint32_t>(hot_table->GetDim(1)));
    data.set_kvRecordElements(
        static_cast<uint32_t>(destination_kv->GetDim(2)));
    data.set_ropeRecordElements(
        static_cast<uint32_t>(destination_rope->GetDim(2)));
    data.set_poolCapacity(static_cast<uint32_t>(pool_capacity));
    data.set_sourcePhysicalBlockCount(
        static_cast<uint32_t>(source_kv->GetDim(0)));
    data.set_destinationPhysicalBlockCount(
        static_cast<uint32_t>(destination_kv->GetDim(0)));
    data.set_jitterEnable(1U);
    data.set_jitterSeed(kJitterSeed);
    const uint64_t pair_count =
        static_cast<uint64_t>(query_num) * 2048U;
    context->SetBlockDim(static_cast<uint32_t>(
        std::min(pair_count, static_cast<uint64_t>(aiv_count))));
    data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct AsuKvGatherDirectV2CompileInfo {};

static ge::graphStatus ParseAsuKvGatherDirectV2(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AsuKvGatherDirectV2)
    .Tiling(AsuKvGatherDirectV2Tiling)
    .TilingParse<AsuKvGatherDirectV2CompileInfo>(
        ParseAsuKvGatherDirectV2);
}  // namespace optiling
