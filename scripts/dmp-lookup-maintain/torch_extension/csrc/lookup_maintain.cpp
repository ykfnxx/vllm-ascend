#include <tuple>

#include <torch/library.h>

#include "ops_common.h"

namespace dmp_lookup_maintain {
namespace {

constexpr int64_t INDEX_CAPACITY = 144 * 1024;
constexpr int64_t TOTAL_SLOT_COUNT = 10 * 1024;
constexpr int64_t FREE_SLOT_COUNT = 2 * 1024;
constexpr int64_t QUERY_SLOT_COUNT = 2 * 1024;
constexpr int64_t FREE_HEAD_STRIDE = 16;

void CheckState(const at::Tensor& index,
                const at::Tensor& slotToIndex,
                const at::Tensor& freeSlots,
                const at::Tensor& freeHead,
                const at::Tensor& reqPoolEntries,
                int64_t reqNum)
{
    TORCH_CHECK(index.dim() == 2 && index.size(1) == INDEX_CAPACITY,
                "index must have shape [pool_rows, 147456]");
    TORCH_CHECK(slotToIndex.dim() == 2 &&
                    slotToIndex.size(0) == index.size(0) &&
                    slotToIndex.size(1) == TOTAL_SLOT_COUNT,
                "slot_to_index must have shape [pool_rows, 10240]");
    TORCH_CHECK(freeSlots.dim() == 2 &&
                    freeSlots.size(0) == index.size(0) &&
                    freeSlots.size(1) == FREE_SLOT_COUNT,
                "free_slots must have shape [pool_rows, 2048]");
    TORCH_CHECK(freeHead.dim() == 2 &&
                    freeHead.size(0) == index.size(0) &&
                    freeHead.size(1) == FREE_HEAD_STRIDE,
                "free_head must have shape [pool_rows, 16]");
    TORCH_CHECK(reqNum > 0 && reqNum <= index.size(0),
                "req_num must be in [1, pool_rows]");
    TORCH_CHECK(reqPoolEntries.dim() == 1 &&
                    reqPoolEntries.size(0) == reqNum,
                "req_pool_entries must have shape [req_num]");
    for (const at::Tensor* tensor :
         {&index, &slotToIndex, &freeSlots, &freeHead, &reqPoolEntries}) {
        TORCH_CHECK(tensor->scalar_type() == at::kInt,
                    "Lookup/Maintain state tensors must be int32");
        TORCH_CHECK(tensor->is_contiguous(),
                    "Lookup/Maintain state tensors must be contiguous");
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> MakeLookupOutputs(
    const at::Tensor& queryIndex)
{
    TORCH_CHECK(queryIndex.dim() == 2 &&
                    queryIndex.size(1) == QUERY_SLOT_COUNT,
                "query_index must have shape [req_num, 2048]");
    TORCH_CHECK(queryIndex.scalar_type() == at::kInt,
                "query_index must be int32");
    TORCH_CHECK(queryIndex.is_contiguous(),
                "query_index must be contiguous");
    return std::make_tuple(at::empty_like(queryIndex),
                           at::empty_like(queryIndex),
                           at::empty_like(queryIndex),
                           at::empty_like(queryIndex),
                           at::empty({queryIndex.size(0), TOTAL_SLOT_COUNT},
                                     queryIndex.options()));
}

void CheckSeqLens(const at::Tensor& seqLens, int64_t reqNum)
{
    TORCH_CHECK(seqLens.dim() == 1 && seqLens.size(0) == reqNum,
                "seq_lens must have shape [req_num]");
    TORCH_CHECK(seqLens.scalar_type() == at::kInt && seqLens.is_contiguous(),
                "seq_lens must be contiguous int32");
}

void CheckNeedsRefill(const at::Tensor& needsRefill, int64_t reqNum)
{
    TORCH_CHECK(needsRefill.dim() == 1 && needsRefill.size(0) == reqNum,
                "needs_refill must have shape [req_num]");
    TORCH_CHECK(needsRefill.scalar_type() == at::kBool &&
                    needsRefill.is_contiguous(),
                "needs_refill must be contiguous bool");
}

void CheckGatherInputs(const at::Tensor& selectionKRope,
                       const at::Tensor& selectionKvCache,
                       const at::Tensor& selectionBlockTable,
                       const at::Tensor& residentTokenIds,
                       const at::Tensor& queryIndex,
                       const at::Tensor& slotOut,
                       const at::Tensor& missOut,
                       const at::Tensor& needsRefill,
                       const at::Tensor& fullKRope,
                       const at::Tensor& fullKvCache,
                       const at::Tensor& fullBlockTable,
                       const at::Tensor& seqLens)
{
    TORCH_CHECK(selectionKRope.dim() == 3 && selectionKvCache.dim() == 3 &&
                    fullKRope.dim() == 3 && fullKvCache.dim() == 3,
                "KV cache tensors must have shape [blocks, block_size, dim]");
    TORCH_CHECK(selectionKRope.scalar_type() == fullKRope.scalar_type() &&
                    selectionKvCache.scalar_type() == fullKvCache.scalar_type() &&
                    selectionKRope.scalar_type() == selectionKvCache.scalar_type(),
                "selection and full KV cache dtypes must match");
    TORCH_CHECK(selectionKRope.scalar_type() == at::kHalf ||
                    selectionKRope.scalar_type() == at::kBFloat16,
                "DMP Lookup KVGather supports float16 and bfloat16");
    const int64_t batchSize = queryIndex.size(0);
    TORCH_CHECK(queryIndex.dim() == 2 && queryIndex.size(1) == QUERY_SLOT_COUNT,
                "query_index must have shape [batch, 2048]");
    TORCH_CHECK(slotOut.sizes() == queryIndex.sizes() &&
                    missOut.sizes() == queryIndex.sizes(),
                "slot_out and miss_out must match query_index");
    TORCH_CHECK(residentTokenIds.dim() == 2 &&
                    residentTokenIds.size(0) == batchSize &&
                    residentTokenIds.size(1) == TOTAL_SLOT_COUNT,
                "resident_token_ids must have shape [batch, 10240]");
    TORCH_CHECK(selectionBlockTable.dim() == 2 &&
                    selectionBlockTable.size(0) == batchSize &&
                    selectionBlockTable.size(1) * selectionKvCache.size(1) ==
                        TOTAL_SLOT_COUNT,
                "selection_block_table must address 10240 slots per row");
    TORCH_CHECK(fullBlockTable.dim() == 2 &&
                    fullBlockTable.size(0) == batchSize,
                "full_block_table batch must match query_index");
    TORCH_CHECK(needsRefill.dim() == 1 && needsRefill.size(0) == batchSize &&
                    needsRefill.scalar_type() == at::kBool,
                "needs_refill must be bool [batch]");
    CheckSeqLens(seqLens, batchSize);
    for (const at::Tensor* tensor : {&selectionBlockTable,
                                     &residentTokenIds,
                                     &queryIndex,
                                     &slotOut,
                                     &missOut,
                                     &fullBlockTable,
                                     &seqLens}) {
        TORCH_CHECK(tensor->scalar_type() == at::kInt,
                    "DMP Lookup KVGather index tensors must be int32");
        TORCH_CHECK(tensor->is_contiguous(),
                    "DMP Lookup KVGather tensors must be contiguous");
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> asuHbmIndexLookupNpu(
    at::Tensor& index,
    at::Tensor& slotToIndex,
    at::Tensor& freeSlots,
    at::Tensor& freeHead,
    const at::Tensor& reqPoolEntries,
    const at::Tensor& queryIndex,
    const at::Tensor& seqLens,
    const at::Tensor& needsRefill,
    int64_t reqNum)
{
    CheckState(index, slotToIndex, freeSlots, freeHead, reqPoolEntries, reqNum);
    TORCH_CHECK(queryIndex.size(0) == reqNum,
                "query_index batch must match req_num");
    CheckSeqLens(seqLens, reqNum);
    CheckNeedsRefill(needsRefill, reqNum);
    auto outputs = MakeLookupOutputs(queryIndex);
    auto& slotOut = std::get<0>(outputs);
    auto& missOut = std::get<1>(outputs);
    auto& hitSparseIndices = std::get<2>(outputs);
    auto& missSparseIndices = std::get<3>(outputs);
    auto& residentTokenIds = std::get<4>(outputs);
    EXEC_NPU_CMD_v0(aclnnAsuHbmIndexLookup,
                    index,
                    slotToIndex,
                    freeSlots,
                    freeHead,
                    reqPoolEntries,
                    queryIndex,
                    seqLens,
                    needsRefill,
                    reqNum,
                    slotOut,
                    missOut,
                    hitSparseIndices,
                    missSparseIndices,
                    residentTokenIds);
    return outputs;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> asuHbmIndexLookupMeta(
    at::Tensor& index,
    at::Tensor& slotToIndex,
    at::Tensor& freeSlots,
    at::Tensor& freeHead,
    const at::Tensor& reqPoolEntries,
    const at::Tensor& queryIndex,
    const at::Tensor& seqLens,
    const at::Tensor& needsRefill,
    int64_t reqNum)
{
    (void)index;
    (void)slotToIndex;
    (void)freeSlots;
    (void)freeHead;
    (void)reqPoolEntries;
    (void)seqLens;
    (void)needsRefill;
    (void)reqNum;
    return MakeLookupOutputs(queryIndex);
}

at::Tensor dmpLookupKvGatherNpu(
    at::Tensor& selectionKRope,
    at::Tensor& selectionKvCache,
    const at::Tensor& selectionBlockTable,
    const at::Tensor& residentTokenIds,
    const at::Tensor& queryIndex,
    const at::Tensor& slotOut,
    const at::Tensor& missOut,
    const at::Tensor& needsRefill,
    const at::Tensor& fullKRope,
    const at::Tensor& fullKvCache,
    const at::Tensor& fullBlockTable,
    const at::Tensor& seqLens)
{
    CheckGatherInputs(selectionKRope,
                      selectionKvCache,
                      selectionBlockTable,
                      residentTokenIds,
                      queryIndex,
                      slotOut,
                      missOut,
                      needsRefill,
                      fullKRope,
                      fullKvCache,
                      fullBlockTable,
                      seqLens);
    at::Tensor copiedCount = at::empty_like(seqLens);
    EXEC_NPU_CMD_v0(aclnnDmpLookupKvGather,
                    selectionKRope,
                    selectionKvCache,
                    selectionBlockTable,
                    residentTokenIds,
                    queryIndex,
                    slotOut,
                    missOut,
                    needsRefill,
                    fullKRope,
                    fullKvCache,
                    fullBlockTable,
                    seqLens,
                    selectionKRope,
                    selectionKvCache,
                    copiedCount);
    return copiedCount;
}

at::Tensor dmpLookupKvGatherMeta(
    at::Tensor& selectionKRope,
    at::Tensor& selectionKvCache,
    const at::Tensor& selectionBlockTable,
    const at::Tensor& residentTokenIds,
    const at::Tensor& queryIndex,
    const at::Tensor& slotOut,
    const at::Tensor& missOut,
    const at::Tensor& needsRefill,
    const at::Tensor& fullKRope,
    const at::Tensor& fullKvCache,
    const at::Tensor& fullBlockTable,
    const at::Tensor& seqLens)
{
    (void)selectionKRope;
    (void)selectionKvCache;
    (void)selectionBlockTable;
    (void)residentTokenIds;
    (void)queryIndex;
    (void)slotOut;
    (void)missOut;
    (void)needsRefill;
    (void)fullKRope;
    (void)fullKvCache;
    (void)fullBlockTable;
    return at::empty_like(seqLens);
}

void asuHbmIndexMaintainNpu(at::Tensor& index,
                            at::Tensor& slotToIndex,
                            at::Tensor& freeSlots,
                            at::Tensor& freeHead,
                            const at::Tensor& reqPoolEntries,
                            const at::Tensor& lastQuerySlots,
                            int64_t reqNum,
                            int64_t seed)
{
    CheckState(index, slotToIndex, freeSlots, freeHead, reqPoolEntries, reqNum);
    TORCH_CHECK(lastQuerySlots.dim() == 2 &&
                    lastQuerySlots.size(0) == reqNum &&
                    lastQuerySlots.size(1) == QUERY_SLOT_COUNT,
                "last_query_slots must have shape [req_num, 2048]");
    TORCH_CHECK(lastQuerySlots.scalar_type() == at::kInt &&
                    lastQuerySlots.is_contiguous(),
                "last_query_slots must be contiguous int32");
    EXEC_NPU_CMD_v0(aclnnAsuHbmIndexMaintainAicpu,
                    index,
                    slotToIndex,
                    freeSlots,
                    freeHead,
                    reqPoolEntries,
                    lastQuerySlots,
                    reqNum,
                    seed,
                    index,
                    slotToIndex,
                    freeSlots,
                    freeHead);
}

void asuHbmIndexMaintainMeta(at::Tensor& index,
                             at::Tensor& slotToIndex,
                             at::Tensor& freeSlots,
                             at::Tensor& freeHead,
                             const at::Tensor& reqPoolEntries,
                             const at::Tensor& lastQuerySlots,
                             int64_t reqNum,
                             int64_t seed)
{
    (void)index;
    (void)slotToIndex;
    (void)freeSlots;
    (void)freeHead;
    (void)reqPoolEntries;
    (void)lastQuerySlots;
    (void)reqNum;
    (void)seed;
}

}  // namespace dmp_lookup_maintain

TORCH_LIBRARY_IMPL(dmp_lookup_maintain, PrivateUse1, m)
{
    m.impl("asu_hbm_index_lookup",
           &dmp_lookup_maintain::asuHbmIndexLookupNpu);
    m.impl("asu_hbm_index_maintain_aicpu",
           &dmp_lookup_maintain::asuHbmIndexMaintainNpu);
    m.impl("dmp_lookup_kv_gather",
           &dmp_lookup_maintain::dmpLookupKvGatherNpu);
}

TORCH_LIBRARY_IMPL(dmp_lookup_maintain, Meta, m)
{
    m.impl("asu_hbm_index_lookup",
           &dmp_lookup_maintain::asuHbmIndexLookupMeta);
    m.impl("asu_hbm_index_maintain_aicpu",
           &dmp_lookup_maintain::asuHbmIndexMaintainMeta);
    m.impl("dmp_lookup_kv_gather",
           &dmp_lookup_maintain::dmpLookupKvGatherMeta);
}
