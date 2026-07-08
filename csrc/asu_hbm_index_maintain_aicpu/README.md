# ASU HBM Index Maintain AICPU

This directory keeps the v0.1 AICPU maintain sources in the vLLM-Ascend
`csrc` operator layout, but this is not the current runnable build path.

The current runnable path is the ASU direct AICPU shared library:

```bash
cd /home/solidyang/workspace/ASU-Ascend/ops
bash build.sh maintain_aicpu Ascend910B3
```

This produces:

```text
/home/solidyang/workspace/ASU-Ascend/ops/build/maintain_aicpu/lib/libasu_hbm_index_maintain_aicpu.so
```

Enable it in vllm-ascend with:

```bash
export VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_AICPU_MAINTAIN_LIB=\
/home/solidyang/workspace/ASU-Ascend/ops/build/maintain_aicpu/lib/libasu_hbm_index_maintain_aicpu.so
unset VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS
```

When this env var is set, `NPUModelRunner` injects a Python `maintain_op`
loaded from `tmp/direct_maintain.py`. `OffloadKVCacheV0Manager._call_maintain`
then calls that injected direct AICPU function instead of falling back to
`torch.ops._C_ascend.asu_hbm_index_maintain_aicpu`.

The `_C_ascend.asu_hbm_index_maintain_aicpu` binding remains in
`csrc/torch_binding.cpp`, but the vLLM-Ascend AICPU custom-op packaging still
needs opdef support before this directory can be used as the normal
`csrc/build.sh -n asu_hbm_index_maintain_aicpu` path.
