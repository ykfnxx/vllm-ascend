#include "asu_kv_resolver_tiling.h"

namespace optiling {
constexpr size_t ORIGINAL_TOPK_INDICES_INDEX = 0;
constexpr size_t ORIGINAL_KV_CACHE_0_INDEX = 8;
constexpr size_t ORIGINAL_KV_CACHE_1_INDEX = 9;
constexpr size_t ACTUAL_SEQ_LEN_ATTR_INDEX = 0;
constexpr size_t BLOCK_SIZE_ATTR_INDEX = 2;

static int64_t ShapeNumel(const gert::StorageShape* shape)
{
    int64_t numel = 1;
    const gert::Shape& storageShape = shape->GetStorageShape();
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        numel *= storageShape.GetDim(i);
    }
    return numel;
}

static int64_t SlotElements(const gert::StorageShape* shape)
{
    const gert::Shape& storageShape = shape->GetStorageShape();
    if (storageShape.GetDimNum() <= 1) {
        return 1;
    }
    int64_t slotNum = storageShape.GetDim(0);
    return ShapeNumel(shape) / slotNum;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* topkShape =
        context->GetInputShape(ORIGINAL_TOPK_INDICES_INDEX);
    const gert::StorageShape* kv0Shape =
        context->GetInputShape(ORIGINAL_KV_CACHE_0_INDEX);
    const gert::StorageShape* kv1Shape =
        context->GetInputShape(ORIGINAL_KV_CACHE_1_INDEX);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();

    AsuKvResolverTilingData tiling;
    tiling.set_topkNumel(ShapeNumel(topkShape));
    tiling.set_actualSeqLen(
        *attrs->GetAttrPointer<int64_t>(ACTUAL_SEQ_LEN_ATTR_INDEX));
    tiling.set_blockSize(
        *attrs->GetAttrPointer<int64_t>(BLOCK_SIZE_ATTR_INDEX));
    tiling.set_kv0SlotElements(SlotElements(kv0Shape));
    tiling.set_kv1SlotElements(SlotElements(kv1Shape));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    context->SetBlockDim(1);
    return ge::GRAPH_SUCCESS;
}

struct AsuKvResolverCompileInfo {};

static ge::graphStatus TilingParseForAsuKvResolver(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AsuKvResolver)
    .Tiling(TilingFunc)
    .TilingParse<AsuKvResolverCompileInfo>(TilingParseForAsuKvResolver);

} // namespace optiling
