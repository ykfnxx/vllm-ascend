#include <tuple>

#include <torch/library.h>

#include "ops_common.h"

namespace custom {

constexpr int64_t DECODE_UPDATE_POOL_SPARSE_COUNT = 2048;
constexpr int64_t DECODE_UPDATE_POOL_CACHE_SLOTS_SIZE = 262144;

std::tuple<at::Tensor, at::Tensor, at::Tensor> MakeLightningIndexerDecodeUpdatePoolOutput(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &reqPoolEntries, const at::Tensor &cacheSlots,
    const at::Tensor &actualSeqLengthsKey, const at::Tensor &blockTable)
{
    TORCH_CHECK(query.dim() == 3, "query must have TND decode shape [B, 64, 128].");
    TORCH_CHECK(key.dim() == 4, "key must have PA_BSND shape [num_blocks, block_size, 1, 128].");
    TORCH_CHECK(weights.dim() == 2, "weights must have shape [B, 64].");
    TORCH_CHECK(reqPoolEntries.dim() == 1, "req_pool_entries must have shape [B].");
    TORCH_CHECK(cacheSlots.dim() == 2, "cache_slots must have shape [pool_size, 262144].");
    TORCH_CHECK(actualSeqLengthsKey.dim() == 1, "actual_seq_lengths_key must be rank 1.");
    TORCH_CHECK(blockTable.dim() == 2, "block_table must be rank 2.");
    TORCH_CHECK(query.size(0) == weights.size(0), "query and weights batch dim must match.");
    TORCH_CHECK(query.size(0) == reqPoolEntries.size(0),
        "req_pool_entries length must match query batch.");
    TORCH_CHECK(query.size(0) == actualSeqLengthsKey.size(0),
        "actual_seq_lengths_key length must match query batch.");
    TORCH_CHECK(query.size(0) == blockTable.size(0), "block_table batch dim must match query batch.");
    TORCH_CHECK(cacheSlots.size(0) > 0 && cacheSlots.size(1) == DECODE_UPDATE_POOL_CACHE_SLOTS_SIZE,
        "cache_slots must have shape [pool_size, 262144] with pool_size > 0.");
    TORCH_CHECK(query.size(1) == 64, "query N1 must be 64.");
    TORCH_CHECK(key.size(2) == 1, "key N2 must be 1.");
    TORCH_CHECK(weights.size(1) == 64, "weights N1 must be 64.");
    TORCH_CHECK(query.size(2) == 128 && key.size(3) == 128, "head_dim must be 128.");
    TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query and key dtype must match.");
    TORCH_CHECK(query.scalar_type() == weights.scalar_type(), "query and weights dtype must match.");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
        "query/key/weights dtype must be fp16 or bf16.");
    TORCH_CHECK(reqPoolEntries.scalar_type() == at::kInt, "req_pool_entries must be int32.");
    TORCH_CHECK(cacheSlots.scalar_type() == at::kInt, "cache_slots must be int32.");
    TORCH_CHECK(actualSeqLengthsKey.scalar_type() == at::kInt, "actual_seq_lengths_key must be int32.");
    TORCH_CHECK(blockTable.scalar_type() == at::kInt, "block_table must be int32.");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous.");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous.");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous.");
    TORCH_CHECK(reqPoolEntries.is_contiguous(), "req_pool_entries must be contiguous.");
    TORCH_CHECK(cacheSlots.is_contiguous(), "cache_slots must be contiguous.");
    TORCH_CHECK(actualSeqLengthsKey.is_contiguous(), "actual_seq_lengths_key must be contiguous.");
    TORCH_CHECK(blockTable.is_contiguous(), "block_table must be contiguous.");

    auto topkIndex = at::empty({query.size(0), key.size(2), DECODE_UPDATE_POOL_SPARSE_COUNT},
        query.options().dtype(at::kInt));
    auto topkSlots = at::empty({query.size(0), key.size(2), DECODE_UPDATE_POOL_SPARSE_COUNT},
        query.options().dtype(at::kInt));
    auto missCount = at::empty({query.size(0)}, query.options().dtype(at::kInt));
    return std::make_tuple(topkIndex, topkSlots, missCount);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_lightning_indexer_decode_update_pool_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &reqPoolEntries, at::Tensor &cacheSlots,
    const at::Tensor &actualSeqLengthsKey, const at::Tensor &blockTable)
{
    auto outputs = MakeLightningIndexerDecodeUpdatePoolOutput(
        query, key, weights, reqPoolEntries, cacheSlots, actualSeqLengthsKey, blockTable);
    auto &topkIndex = std::get<0>(outputs);
    auto &topkSlots = std::get<1>(outputs);
    auto &missCount = std::get<2>(outputs);
    EXEC_NPU_CMD_v0(aclnnLightningIndexerDecodeUpdatePool, query, key, weights, reqPoolEntries, cacheSlots,
        actualSeqLengthsKey, blockTable, topkIndex, topkSlots, missCount);
    return outputs;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_lightning_indexer_decode_update_pool_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &reqPoolEntries, at::Tensor &cacheSlots,
    const at::Tensor &actualSeqLengthsKey, const at::Tensor &blockTable)
{
    return MakeLightningIndexerDecodeUpdatePoolOutput(
        query, key, weights, reqPoolEntries, cacheSlots, actualSeqLengthsKey, blockTable);
}

} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_lightning_indexer_decode_update_pool", &custom::npu_lightning_indexer_decode_update_pool_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_lightning_indexer_decode_update_pool", &custom::npu_lightning_indexer_decode_update_pool_meta);
}
