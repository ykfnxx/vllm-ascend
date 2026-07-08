# ASU direct AICPU maintain debug path

This directory keeps a temporary eager-mode path for calling the ASU direct
AICPU maintain shared library from vllm-ascend without packaging it as a
vllm-ascend custom op.

In the current framework bring-up:

- lookup is compiled and called as the vllm-ascend custom op
  `torch.ops._C_ascend.asu_hbm_index_lookup`.
- maintain is compiled from ASU-Ascend as a direct AICPU `.so` and injected as
  `OffloadKVCacheV0Manager.maintain_op`.

Build the ASU direct AICPU library from the ASU-Ascend repo:

```bash
cd /home/solidyang/workspace/ASU-Ascend/ops
bash build.sh maintain_aicpu Ascend910B3
```

Then run vllm-ascend with:

```bash
export VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB=/path/to/libasu_hbm_index_maintain_aicpu.so
unset VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS
```

With this env var set:

- lookup still uses the real `torch.ops._C_ascend.asu_hbm_index_lookup` op.
- maintain uses `ctypes` to call `asu_hbm_index_maintain_do` from the ASU
  direct AICPU `.so`.
- `VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS=1` overrides this path and switches
  both lookup and maintain to the Python reference implementation.
- the Python wrapper is loaded from the source tree, so this is not a wheel
  packaging path.
- this path is for eager-mode bring-up only and is not graph-compatible.

The expected C ABI is:

```c
void asu_hbm_index_maintain_do(
    uint32_t blockDim,
    void* stream,
    void* index,
    void* slotToIndex,
    void* freeSlots,
    void* freeHead,
    void* lastQuerySlots,
    uint32_t reqNum,
    uint32_t seed);
```
