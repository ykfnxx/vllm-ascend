#include <torch/library.h>

#include "ops_common.h"

namespace custom {

constexpr int64_t DECODE_SPARSE_COUNT = 2048;

at::Tensor MakeLightningIndexerDecodeOutput(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &actualSeqLengthsKey, const at::Tensor &blockTable)
{
    TORCH_CHECK(query.dim() == 3, "query must have TND decode shape [B, 64, 128].");
    TORCH_CHECK(key.dim() == 4, "key must have PA_BSND shape [num_blocks, block_size, 1, 128].");
    TORCH_CHECK(weights.dim() == 2, "weights must have shape [B, 64].");
    TORCH_CHECK(actualSeqLengthsKey.dim() == 1, "actual_seq_lengths_key must be rank 1.");
    TORCH_CHECK(blockTable.dim() == 2, "block_table must be rank 2.");
    TORCH_CHECK(query.size(0) == weights.size(0), "query and weights batch dim must match.");
    TORCH_CHECK(query.size(0) == actualSeqLengthsKey.size(0),
        "actual_seq_lengths_key length must match query batch.");
    TORCH_CHECK(query.size(0) == blockTable.size(0), "block_table batch dim must match query batch.");
    TORCH_CHECK(query.size(1) == 64, "query N1 must be 64.");
    TORCH_CHECK(key.size(2) == 1, "key N2 must be 1.");
    TORCH_CHECK(weights.size(1) == 64, "weights N1 must be 64.");
    TORCH_CHECK(query.size(2) == 128 && key.size(3) == 128, "head_dim must be 128.");
    TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query and key dtype must match.");
    TORCH_CHECK(query.scalar_type() == weights.scalar_type(), "query and weights dtype must match.");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
        "query/key/weights dtype must be fp16 or bf16.");
    TORCH_CHECK(actualSeqLengthsKey.scalar_type() == at::kInt, "actual_seq_lengths_key must be int32.");
    TORCH_CHECK(blockTable.scalar_type() == at::kInt, "block_table must be int32.");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous.");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous.");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous.");
    TORCH_CHECK(actualSeqLengthsKey.is_contiguous(), "actual_seq_lengths_key must be contiguous.");
    TORCH_CHECK(blockTable.is_contiguous(), "block_table must be contiguous.");

    return at::empty({query.size(0), key.size(2), DECODE_SPARSE_COUNT}, query.options().dtype(at::kInt));
}

at::Tensor npu_lightning_indexer_decode_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &actualSeqLengthsKey, const at::Tensor &blockTable)
{
    auto sparseIndices = MakeLightningIndexerDecodeOutput(query, key, weights, actualSeqLengthsKey, blockTable);
    EXEC_NPU_CMD_v0(aclnnLightningIndexerDecode, query, key, weights, actualSeqLengthsKey, blockTable, sparseIndices);
    return sparseIndices;
}

at::Tensor npu_lightning_indexer_decode_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &actualSeqLengthsKey, const at::Tensor &blockTable)
{
    return MakeLightningIndexerDecodeOutput(query, key, weights, actualSeqLengthsKey, blockTable);
}

} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_lightning_indexer_decode", &custom::npu_lightning_indexer_decode_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_lightning_indexer_decode", &custom::npu_lightning_indexer_decode_meta);
}
