# ASU HBM Index Lookup Custom Op

This directory contains the v0.1 lookup operator in the vLLM-Ascend `csrc`
custom-op layout.

Build and install only this OPP on an Ascend/CANN host:

```bash
cd /home/solidyang/workspace/vllm-ascend/csrc

bash build.sh -n asu_hbm_index_lookup -c ascend910b

./output/CANN-custom_ops*.run \
  --install-path=/home/solidyang/workspace/vllm-ascend/vllm_ascend/_cann_ops_custom
```

Use `-c ascend910_93` on A3.

The Python callable is registered by `vllm_ascend_C`, so also build
vllm-ascend with custom kernels enabled:

```bash
cd /home/solidyang/workspace/vllm-ascend

export COMPILE_CUSTOM_KERNELS=1
python3 -m pip install --no-build-isolation -e .
```

If `csrc/build_aclnn.sh` still does not include `asu_hbm_index_lookup` in the
default A2/A3 custom-op list, rerun the single-op `csrc/build.sh` command after
`pip install -e .` so the lookup OPP is present under
`vllm_ascend/_cann_ops_custom`.

Runtime calls use:

```text
torch.ops._C_ascend.asu_hbm_index_lookup(...)
```

In the offload framework this is called from
`OffloadKVCacheV0Manager._call_lookup()` when `lookup_op` is `None`.
