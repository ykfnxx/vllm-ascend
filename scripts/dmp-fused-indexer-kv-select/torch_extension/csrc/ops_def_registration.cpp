#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(custom, m)
{
    m.def(
        "npu_lightning_indexer_decode(Tensor query, Tensor key, Tensor weights, "
        "Tensor actual_seq_lengths_key, Tensor block_table) -> Tensor");
    m.def(
        "npu_lightning_indexer_decode_update(Tensor query, Tensor key, Tensor weights, Tensor(a!) cache_slots, "
        "Tensor actual_seq_lengths_key, Tensor block_table) -> (Tensor, Tensor, Tensor)");
    m.def(
        "npu_lightning_indexer_decode_update_pool(Tensor query, Tensor key, Tensor weights, "
        "Tensor req_pool_entries, Tensor(a!) cache_slots, Tensor actual_seq_lengths_key, "
        "Tensor block_table) -> (Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
