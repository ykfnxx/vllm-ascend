# DsaSparseLookupUpdate standalone profiling build

This directory builds the existing registered `DsaSparseLookupUpdate` operator
without including the vllm-ascend CMake project or PyTorch extension. The
copied `op_host`, `op_kernel`, and ACLNN declarations retain the production
operator structure. The standalone CMake layer uses CANN `npu_op_*` APIs and
adds `-g` to the optimized Ascend C kernel build for msOpProf source and
operand-record analysis.

Build, install, and compile the independent ACLNN runner:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash tools/dsa_sparse_lookup_update/standalone/build.sh --clean
```

The default isolated install root is `standalone/.install`; no system OPP
directory is modified. Run the one-shot workload directly with:

```bash
source tools/dsa_sparse_lookup_update/standalone/.install/vendors/dsa_sparse_prof/bin/set_env.bash
tools/dsa_sparse_lookup_update/standalone/build_runner/dsa_sparse_lookup_update_runner \
  --device npu:0 --requests 32 --miss-rate 10
```

Collect Roofline data through the repository wrapper:

```bash
bash tools/dsa_sparse_lookup_update/profile_roofline.sh \
  --device npu:0 --requests 32 --miss-rate 10
```
