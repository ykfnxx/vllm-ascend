#include "asu_kv_gather_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t DESTINATION_KV_INPUT = 0U;
constexpr uint32_t DESTINATION_ROPE_INPUT = 1U;
constexpr uint32_t DESTINATION_BLOCK_TABLE_INPUT = 2U;
constexpr uint32_t SOURCE_KV_INPUT = 3U;
constexpr uint32_t SOURCE_ROPE_INPUT = 4U;
constexpr uint32_t SOURCE_BLOCK_TABLE_INPUT = 5U;
constexpr uint32_t REQ_POOL_ENTRIES_INPUT = 6U;
constexpr uint32_t TOKEN_POSITIONS_INPUT = 7U;
constexpr uint32_t DESTINATION_SLOTS_INPUT = 8U;
constexpr uint32_t MISS_MASK_INPUT = 9U;
constexpr uint32_t ATTR_BLOCK_SIZE = 0U;
constexpr uint32_t ATTR_REQ_NUM = 1U;

constexpr uint32_t kSwapJitterSeed = 0x9E3779B9U;

bool IsPositiveUint32(int64_t value)
{
    return value > 0 &&
           static_cast<uint64_t>(value) <=
               std::numeric_limits<uint32_t>::max();
}

bool FitsUint64Product(int64_t first, int64_t second, int64_t third)
{
    const uint64_t first_value = static_cast<uint64_t>(first);
    const uint64_t second_value = static_cast<uint64_t>(second);
    const uint64_t third_value = static_cast<uint64_t>(third);
    const uint64_t limit = std::numeric_limits<uint64_t>::max();
    return first_value <= limit / second_value &&
           first_value * second_value <= limit / third_value;
}

bool IsSupportedDtypePair(ge::DataType kv_dtype, ge::DataType rope_dtype)
{
    return (kv_dtype == ge::DT_FLOAT16 && rope_dtype == ge::DT_FLOAT16) ||
           (kv_dtype == ge::DT_BF16 && rope_dtype == ge::DT_BF16) ||
           (kv_dtype == ge::DT_INT8 && rope_dtype == ge::DT_BF16);
}

uint32_t GetDataTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
            return sizeof(int8_t);
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return sizeof(uint16_t);
        default:
            return 0U;
    }
}
}  // namespace

static ge::graphStatus AsuKvGatherTilingFunc(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("AsuKvGather", "TilingContext is nullptr."),
               return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* block_size =
        attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE);
    const int64_t* req_num = attrs->GetAttrPointer<int64_t>(ATTR_REQ_NUM);
    OPS_LOG_E_IF_NULL(context, block_size, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, req_num, return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsPositiveUint32(*block_size) ||
                   !IsPositiveUint32(*req_num),
               OPS_LOG_E(context->GetNodeName(),
                         "block_size and req_num must be positive uint32 values."),
               return ge::GRAPH_FAILED);

    const gert::StorageShape* destination_kv_shape =
        context->GetInputShape(DESTINATION_KV_INPUT);
    const gert::StorageShape* destination_rope_shape =
        context->GetInputShape(DESTINATION_ROPE_INPUT);
    const gert::StorageShape* destination_table_shape =
        context->GetInputShape(DESTINATION_BLOCK_TABLE_INPUT);
    const gert::StorageShape* source_kv_shape =
        context->GetInputShape(SOURCE_KV_INPUT);
    const gert::StorageShape* source_rope_shape =
        context->GetInputShape(SOURCE_ROPE_INPUT);
    const gert::StorageShape* source_table_shape =
        context->GetInputShape(SOURCE_BLOCK_TABLE_INPUT);
    const gert::StorageShape* req_pool_shape =
        context->GetInputShape(REQ_POOL_ENTRIES_INPUT);
    const gert::StorageShape* token_shape =
        context->GetInputShape(TOKEN_POSITIONS_INPUT);
    const gert::StorageShape* destination_slots_shape =
        context->GetInputShape(DESTINATION_SLOTS_INPUT);
    const gert::StorageShape* miss_mask_shape =
        context->GetInputShape(MISS_MASK_INPUT);
    OPS_LOG_E_IF_NULL(context, destination_kv_shape,
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, destination_rope_shape,
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, destination_table_shape,
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, source_kv_shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, source_rope_shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, source_table_shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, req_pool_shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, token_shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, destination_slots_shape,
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, miss_mask_shape, return ge::GRAPH_FAILED);

    const gert::Shape& destination_kv_storage =
        destination_kv_shape->GetStorageShape();
    const gert::Shape& destination_rope_storage =
        destination_rope_shape->GetStorageShape();
    const gert::Shape& destination_table_storage =
        destination_table_shape->GetStorageShape();
    const gert::Shape& source_kv_storage =
        source_kv_shape->GetStorageShape();
    const gert::Shape& source_rope_storage =
        source_rope_shape->GetStorageShape();
    const gert::Shape& source_table_storage =
        source_table_shape->GetStorageShape();
    const gert::Shape& req_pool_storage =
        req_pool_shape->GetStorageShape();
    const gert::Shape& token_storage = token_shape->GetStorageShape();
    const gert::Shape& destination_slots_storage =
        destination_slots_shape->GetStorageShape();
    const gert::Shape& miss_mask_storage =
        miss_mask_shape->GetStorageShape();
    OPS_ERR_IF(destination_kv_storage.GetDimNum() != 3U ||
               destination_rope_storage.GetDimNum() != 3U ||
               source_kv_storage.GetDimNum() != 3U ||
               source_rope_storage.GetDimNum() != 3U ||
               destination_table_storage.GetDimNum() != 2U ||
               source_table_storage.GetDimNum() != 2U ||
               req_pool_storage.GetDimNum() != 1U ||
               token_storage.GetDimNum() != 2U ||
               destination_slots_storage.GetDimNum() != 2U ||
               miss_mask_storage.GetDimNum() != 2U,
               OPS_LOG_E(context->GetNodeName(), "Invalid input rank."),
               return ge::GRAPH_FAILED);

    const int64_t request_count = *req_num;
    const int64_t query_count = token_storage.GetDim(1);
    OPS_ERR_IF(destination_table_storage.GetDim(0) != request_count ||
               req_pool_storage.GetDim(0) != request_count ||
               miss_mask_storage.GetDim(0) != request_count,
               OPS_LOG_E(context->GetNodeName(),
                         "Request-row tensors must match req_num."),
               return ge::GRAPH_FAILED);
    const bool dense_layout =
        token_storage.GetDim(0) == request_count &&
        destination_slots_storage.GetDim(0) == request_count &&
        miss_mask_storage.GetDim(0) == request_count &&
        destination_slots_storage.GetDim(1) == query_count &&
        miss_mask_storage.GetDim(1) == query_count;
    const bool resident_init_layout =
        token_storage.GetDim(0) == 1 &&
        destination_slots_storage.GetDim(0) == 1 &&
        destination_slots_storage.GetDim(1) == query_count &&
        miss_mask_storage.GetDim(0) == request_count &&
        miss_mask_storage.GetDim(1) == 1;
    OPS_ERR_IF(!dense_layout && !resident_init_layout,
               OPS_LOG_E(
                   context->GetNodeName(),
                   "Index metadata must use dense [requests, query_count] "
                   "layout or resident initialization [1, query_count], "
                   "[1, query_count], [requests, 1] layout."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsPositiveUint32(query_count) ||
               !IsPositiveUint32(source_table_storage.GetDim(0)) ||
               !IsPositiveUint32(source_table_storage.GetDim(1)) ||
               !IsPositiveUint32(destination_table_storage.GetDim(1)) ||
               !IsPositiveUint32(source_kv_storage.GetDim(0)) ||
               !IsPositiveUint32(destination_kv_storage.GetDim(0)) ||
               !IsPositiveUint32(destination_rope_storage.GetDim(2)) ||
               !IsPositiveUint32(destination_rope_storage.GetDim(2)),
               OPS_LOG_E(context->GetNodeName(),
                         "ASU gather capacities and record dimensions must be "
                         "positive uint32 values."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(destination_kv_storage.GetDim(1) != *block_size ||
               destination_rope_storage.GetDim(1) != *block_size ||
               source_kv_storage.GetDim(1) != *block_size ||
               source_rope_storage.GetDim(1) != *block_size,
               OPS_LOG_E(context->GetNodeName(),
                         "All cache block dimensions must equal block_size."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(destination_kv_storage.GetDim(0) !=
                   destination_rope_storage.GetDim(0) ||
               source_kv_storage.GetDim(0) !=
                   source_rope_storage.GetDim(0),
               OPS_LOG_E(context->GetNodeName(),
                         "KV and RoPE physical block counts must match."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(destination_kv_storage.GetDim(2) !=
                   source_kv_storage.GetDim(2) ||
               destination_rope_storage.GetDim(2) !=
                   source_rope_storage.GetDim(2),
               OPS_LOG_E(context->GetNodeName(),
                         "Source and destination record dimensions must match."),
               return ge::GRAPH_FAILED);

    const auto* destination_kv_desc =
        context->GetInputDesc(DESTINATION_KV_INPUT);
    const auto* destination_rope_desc =
        context->GetInputDesc(DESTINATION_ROPE_INPUT);
    const auto* destination_table_desc =
        context->GetInputDesc(DESTINATION_BLOCK_TABLE_INPUT);
    const auto* source_kv_desc = context->GetInputDesc(SOURCE_KV_INPUT);
    const auto* source_rope_desc = context->GetInputDesc(SOURCE_ROPE_INPUT);
    const auto* source_table_desc =
        context->GetInputDesc(SOURCE_BLOCK_TABLE_INPUT);
    const auto* req_pool_desc =
        context->GetInputDesc(REQ_POOL_ENTRIES_INPUT);
    const auto* token_desc = context->GetInputDesc(TOKEN_POSITIONS_INPUT);
    const auto* destination_slots_desc =
        context->GetInputDesc(DESTINATION_SLOTS_INPUT);
    const auto* miss_mask_desc = context->GetInputDesc(MISS_MASK_INPUT);
    OPS_LOG_E_IF_NULL(context, destination_kv_desc,
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, destination_rope_desc,
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, destination_table_desc,
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, source_kv_desc, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, source_rope_desc, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, source_table_desc, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, req_pool_desc, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, token_desc, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, destination_slots_desc,
                      return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, miss_mask_desc, return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        static_cast<ge::Format>(
            ge::GetPrimaryFormat(destination_kv_desc->GetStorageFormat())) !=
                ge::FORMAT_ND ||
            static_cast<ge::Format>(
                ge::GetPrimaryFormat(destination_rope_desc->GetStorageFormat())) != ge::FORMAT_ND ||
            static_cast<ge::Format>(
                ge::GetPrimaryFormat(destination_table_desc->GetStorageFormat())) !=
                ge::FORMAT_ND ||
            static_cast<ge::Format>(
                ge::GetPrimaryFormat(source_kv_desc->GetStorageFormat())) !=
                ge::FORMAT_ND ||
            static_cast<ge::Format>(
                ge::GetPrimaryFormat(source_rope_desc->GetStorageFormat())) !=
                ge::FORMAT_ND ||
            static_cast<ge::Format>(
                ge::GetPrimaryFormat(req_pool_desc->GetStorageFormat())) !=
                ge::FORMAT_ND ||
            static_cast<ge::Format>(
                ge::GetPrimaryFormat(token_desc->GetStorageFormat())) !=
                ge::FORMAT_ND ||
            static_cast<ge::Format>(
                ge::GetPrimaryFormat(destination_slots_desc->GetStorageFormat())) !=
                ge::FORMAT_ND ||
            static_cast<ge::Format>(
                ge::GetPrimaryFormat(miss_mask_desc->GetStorageFormat())) !=
                ge::FORMAT_ND,
        OPS_LOG_E(context->GetNodeName(),
                  "All ASU gather tensors must use contiguous ND storage."),
        return ge::GRAPH_FAILED);

    const ge::DataType kv_dtype = destination_kv_desc->GetDataType();
    const ge::DataType rope_dtype = destination_rope_desc->GetDataType();
    OPS_ERR_IF(!IsSupportedDtypePair(kv_dtype, rope_dtype) ||
               source_kv_desc->GetDataType() != kv_dtype ||
               source_rope_desc->GetDataType() != rope_dtype,
               OPS_LOG_E(context->GetNodeName(),
                         "Unsupported or inconsistent KV/RoPE dtypes."),
               return ge::GRAPH_FAILED);
    const uint64_t kv_record_bytes =
        static_cast<uint64_t>(destination_kv_storage.GetDim(2)) *
        GetDataTypeSize(kv_dtype);
    const uint64_t rope_record_bytes =
        static_cast<uint64_t>(destination_rope_storage.GetDim(2)) *
        GetDataTypeSize(rope_dtype);
    OPS_ERR_IF(kv_record_bytes % 32U != 0U ||
               rope_record_bytes % 32U != 0U,
               OPS_LOG_E(context->GetNodeName(),
                         "KV and RoPE records must be 32-byte aligned."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        !FitsUint64Product(source_kv_storage.GetDim(0),
                           *block_size,
                           source_kv_storage.GetDim(2)) ||
            !FitsUint64Product(source_rope_storage.GetDim(0),
                               *block_size,
                               source_rope_storage.GetDim(2)) ||
            !FitsUint64Product(destination_kv_storage.GetDim(0),
                               *block_size,
                               destination_kv_storage.GetDim(2)) ||
            !FitsUint64Product(destination_rope_storage.GetDim(0),
                               *block_size,
                               destination_rope_storage.GetDim(2)),
        OPS_LOG_E(context->GetNodeName(),
                  "Cache element address range exceeds uint64."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(destination_table_desc->GetDataType() != ge::DT_INT32 ||
               source_table_desc->GetDataType() != ge::DT_INT32 ||
               req_pool_desc->GetDataType() != ge::DT_INT32 ||
               token_desc->GetDataType() != ge::DT_INT32 ||
               destination_slots_desc->GetDataType() != ge::DT_INT32 ||
               miss_mask_desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(context->GetNodeName(),
                         "All ASU gather index tensors must be int32."),
               return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platform_info = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platform_info, return ge::GRAPH_FAILED);
    auto platform = platform_ascendc::PlatformAscendC(platform_info);
    const uint32_t aiv_num = platform.GetCoreNumAiv();
    OPS_ERR_IF(aiv_num == 0U,
               OPS_LOG_E(context->GetNodeName(), "AIV core count is 0."),
               return ge::GRAPH_FAILED);
    uint64_t ub_size = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    OPS_ERR_IF(kv_record_bytes + rope_record_bytes > ub_size,
               OPS_LOG_E(context->GetNodeName(), "KV and RoPE record buffers exceed platform UB."),
               return ge::GRAPH_FAILED);

    AsuKvGatherTilingData tiling;
    tiling.set_reqNum(static_cast<uint32_t>(*req_num));
    tiling.set_queryCount(static_cast<uint32_t>(query_count));
    tiling.set_residentInitLayout(resident_init_layout ? 1U : 0U);
    tiling.set_blockSize(static_cast<uint32_t>(*block_size));
    tiling.set_sourceTableWidth(
        static_cast<uint32_t>(source_table_storage.GetDim(1)));
    tiling.set_destinationTableWidth(
        static_cast<uint32_t>(destination_table_storage.GetDim(1)));
    tiling.set_kvRecordElements(
        static_cast<uint32_t>(destination_kv_storage.GetDim(2)));
    tiling.set_ropeRecordElements(
        static_cast<uint32_t>(destination_rope_storage.GetDim(2)));
    tiling.set_sourcePoolCapacity(
        static_cast<uint32_t>(source_table_storage.GetDim(0)));
    tiling.set_sourcePhysicalBlockCount(
        static_cast<uint32_t>(source_kv_storage.GetDim(0)));
    tiling.set_destinationPhysicalBlockCount(
        static_cast<uint32_t>(destination_kv_storage.GetDim(0)));
    // Swap 延迟抖动使用固定的基线延迟和随机种子。
    tiling.set_jitterEnable(1U);
    tiling.set_jitterSeed(kSwapJitterSeed);
    const uint64_t pair_count =
        static_cast<uint64_t>(*req_num) *
        static_cast<uint64_t>(query_count);
    context->SetBlockDim(static_cast<uint32_t>(
        std::min(pair_count, static_cast<uint64_t>(aiv_num))));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct AsuKvGatherCompileInfo {};

static ge::graphStatus TilingParseForAsuKvGather(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AsuKvGather)
    .Tiling(AsuKvGatherTilingFunc)
    .TilingParse<AsuKvGatherCompileInfo>(TilingParseForAsuKvGather);
}  // namespace optiling
