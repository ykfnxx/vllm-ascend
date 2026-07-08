# ASU HBM Index Lookup

This directory keeps the v0.1 lookup custom-op sources in the vLLM-Ascend
`csrc` operator layout, but the current bring-up path uses the ASU direct
lookup shared library instead of the vLLM-Ascend OPP package.

Build the direct lookup library from ASU-Ascend:

```bash
cd /home/solidyang/workspace/ASU-Ascend/ops
bash build.sh lookup_aiv Ascend910B3
```

Find the generated library:

```bash
find /home/solidyang/workspace/ASU-Ascend/ops/build/lookup_aiv \
  -name 'libasu_hbm_index_lookup_aiv*.so' -print
```

Enable it in vllm-ascend with:

```bash
export VLLM_ASCEND_KV_OFFLOAD_V0_DIRECT_LOOKUP_LIB=\
/home/solidyang/workspace/ASU-Ascend/ops/build/lookup_aiv/lib/libasu_hbm_index_lookup_aiv.so
unset VLLM_ASCEND_KV_OFFLOAD_V0_REF_HBM_OPS
```

When this env var is set, `NPUModelRunner` injects a Python `lookup_op` loaded
from `tmp/direct_lookup.py`. The injected callable keeps the current framework
signature:

```text
lookup_op(index, slot_to_index, free_slots, free_head, query_index, req_num) -> slot_out
```

`OffloadKVCacheV0Manager._call_lookup()` does not need a separate code path; it
calls the injected callable through the same `lookup_op` hook.

The `_C_ascend.asu_hbm_index_lookup` binding and OPP sources remain in this
directory for the future packaged custom-op path.
