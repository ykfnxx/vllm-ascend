#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(dmp_lookup_maintain, m)
{
    m.def(
        "asu_hbm_index_lookup(Tensor(a!) index, Tensor(b!) slot_to_index, "
        "Tensor(c!) free_slots, Tensor(d!) free_head, Tensor req_pool_entries, "
        "Tensor query_index, Tensor seq_lens, Tensor needs_refill, int req_num) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "asu_hbm_index_maintain_aicpu(Tensor(a!) index, "
        "Tensor(b!) slot_to_index, Tensor(c!) free_slots, "
        "Tensor(d!) free_head, Tensor req_pool_entries, "
        "Tensor last_query_slots, int req_num, int seed) -> ()");
    m.def(
        "dmp_lookup_kv_gather(Tensor(a!) selection_k_rope, "
        "Tensor(b!) selection_kv_cache, Tensor selection_block_table, "
        "Tensor resident_token_ids, Tensor query_index, Tensor slot_out, "
        "Tensor miss_out, Tensor needs_refill, Tensor full_k_rope, "
        "Tensor full_kv_cache, Tensor full_block_table, Tensor seq_lens) "
        "-> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
