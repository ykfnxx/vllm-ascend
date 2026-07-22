#include "dmp_lookup_kv_gather_tiling.h"

#include <algorithm>

#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t SELECTION_K_ROPE = 0U;
constexpr uint32_t SELECTION_KV_CACHE = 1U;
constexpr uint32_t SELECTION_BLOCK_TABLE = 2U;
constexpr uint32_t RESIDENT_TOKEN_IDS = 3U;
constexpr uint32_t QUERY_INDEX = 4U;
constexpr uint32_t FULL_K_ROPE = 8U;
constexpr uint32_t FULL_KV_CACHE = 9U;
constexpr uint32_t FULL_BLOCK_TABLE = 10U;
constexpr uint32_t TOTAL_SLOTS = 10U * 1024U;
constexpr uint32_t QUERY_SLOTS = 2U * 1024U;

bool GetShape(gert::TilingContext* context, uint32_t index, gert::Shape& shape)
{
    const auto* input = context->GetInputShape(index);
    if (input == nullptr) {
        return false;
    }
    shape = input->GetStorageShape();
    return true;
}
}  // namespace

static ge::graphStatus DmpLookupKvGatherTilingFunc(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("DmpLookupKvGather", "TilingContext is nullptr."),
               return ge::GRAPH_FAILED);

    gert::Shape selRope;
    gert::Shape selKv;
    gert::Shape selTable;
    gert::Shape resident;
    gert::Shape query;
    gert::Shape fullRope;
    gert::Shape fullKv;
    gert::Shape fullTable;
    OPS_ERR_IF(!GetShape(context, SELECTION_K_ROPE, selRope) ||
                   !GetShape(context, SELECTION_KV_CACHE, selKv) ||
                   !GetShape(context, SELECTION_BLOCK_TABLE, selTable) ||
                   !GetShape(context, RESIDENT_TOKEN_IDS, resident) ||
                   !GetShape(context, QUERY_INDEX, query) ||
                   !GetShape(context, FULL_K_ROPE, fullRope) ||
                   !GetShape(context, FULL_KV_CACHE, fullKv) ||
                   !GetShape(context, FULL_BLOCK_TABLE, fullTable),
               OPS_LOG_E(context->GetNodeName(), "An input shape is unavailable."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(selRope.GetDimNum() != 3 || selKv.GetDimNum() != 3 ||
                   fullRope.GetDimNum() != 3 || fullKv.GetDimNum() != 3 ||
                   selTable.GetDimNum() != 2 || resident.GetDimNum() != 2 ||
                   query.GetDimNum() != 2 || fullTable.GetDimNum() != 2,
               OPS_LOG_E(context->GetNodeName(), "Unexpected Lookup KVGather tensor rank."),
               return ge::GRAPH_FAILED);

    const uint32_t batch = static_cast<uint32_t>(query.GetDim(0));
    const uint32_t selection_block_size = static_cast<uint32_t>(selKv.GetDim(1));
    const uint32_t full_block_size = static_cast<uint32_t>(fullKv.GetDim(1));
    const uint32_t kv_dim = static_cast<uint32_t>(selKv.GetDim(2));
    const uint32_t rope_dim = static_cast<uint32_t>(selRope.GetDim(2));
    OPS_ERR_IF(batch == 0 || query.GetDim(1) != QUERY_SLOTS ||
                   resident.GetDim(0) != batch || resident.GetDim(1) != TOTAL_SLOTS ||
                   selTable.GetDim(0) != batch || fullTable.GetDim(0) != batch ||
                   selRope.GetDim(1) != selection_block_size ||
                   fullRope.GetDim(1) != full_block_size ||
                   fullKv.GetDim(2) != kv_dim || fullRope.GetDim(2) != rope_dim ||
                   selTable.GetDim(1) * selection_block_size != TOTAL_SLOTS,
               OPS_LOG_E(context->GetNodeName(), "Lookup KVGather shapes are incompatible."),
               return ge::GRAPH_FAILED);

    const ge::DataType dtype = context->GetInputDesc(SELECTION_KV_CACHE)->GetDataType();
    const uint32_t element_size = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2U : 0U;
    OPS_ERR_IF(element_size == 0 || (kv_dim * element_size) % 32U != 0U ||
                   (rope_dim * element_size) % 32U != 0U,
               OPS_LOG_E(context->GetNodeName(), "KV and rope rows must be 32-byte aligned."),
               return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platform_info = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platform_info, return ge::GRAPH_FAILED);
    auto platform = platform_ascendc::PlatformAscendC(platform_info);
    const uint32_t aiv_num = platform.GetCoreNumAiv();
    OPS_ERR_IF(aiv_num == 0,
               OPS_LOG_E(context->GetNodeName(), "AIV core count is 0."),
               return ge::GRAPH_FAILED);

    DmpLookupKvGatherTilingData tiling;
    tiling.set_batchSize(batch);
    tiling.set_selectionBlockSize(selection_block_size);
    tiling.set_selectionBlocksPerRow(static_cast<uint32_t>(selTable.GetDim(1)));
    tiling.set_fullBlockSize(full_block_size);
    tiling.set_fullBlocksPerRow(static_cast<uint32_t>(fullTable.GetDim(1)));
    tiling.set_kvDim(kv_dim);
    tiling.set_ropeDim(rope_dim);
    context->SetBlockDim(std::min(batch, aiv_num));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct DmpLookupKvGatherCompileInfo {};

static ge::graphStatus TilingParseForDmpLookupKvGather(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DmpLookupKvGather)
    .Tiling(DmpLookupKvGatherTilingFunc)
    .TilingParse<DmpLookupKvGatherCompileInfo>(TilingParseForDmpLookupKvGather);
}  // namespace optiling
