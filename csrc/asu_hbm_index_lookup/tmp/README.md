# ASU direct lookup debug path

This directory keeps the direct Python wrapper for the ASU lookup shared
library. It is used to downgrade lookup from the vLLM-Ascend custom-op package
to a direct eager-mode call while preserving the current framework call format.

Build the ASU lookup library:

```bash
cd /home/solidyang/workspace/ASU-Ascend/ops
bash build.sh lookup_aiv Ascend910B3
```

Then run vllm-ascend with:

```bash
export VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_LOOKUP_LIB=/path/to/libasu_hbm_index_lookup_aiv.so
unset VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS
```

The wrapper exposes:

```text
lookup_op(index, slot_to_index, free_slots, free_head, query_index, req_num) -> slot_out
```

The expected C ABI is:

```c
void asu_hbm_index_lookup_do(
    uint32_t blockDim,
    void* stream,
    void* index,
    void* slotToIndex,
    void* freeSlots,
    void* freeHead,
    void* queryIndex,
    void* slotOut,
    uint32_t reqNum);
```
