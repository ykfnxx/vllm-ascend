# ASU HBM Index Maintain AICPU Custom Op

This directory contains the v0.1 AICPU maintain custom-op sources in the vLLM-Ascend `csrc` operator layout.

Build on an Ascend/CANN host:

```bash
bash csrc/build.sh -n asu_hbm_index_maintain_aicpu
```

The operator is discovered through:

```text
csrc/asu_hbm_index_maintain_aicpu/op_host/CMakeLists.txt
```

Runtime calls use:

```text
torch.ops._C_ascend.asu_hbm_index_maintain_aicpu(...)
```

The public op name uses the `_aicpu` suffix to avoid binding v0.1 to the AICore `asu_hbm_index_maintain` reference implementation.
