# MockKVSelect AICPU Operator

This directory contains a minimal CANN custom AICPU operator used to measure
runtime scheduling and pipeline overlap cost for a KVSelect-shaped operator.

`MockKVSelect` intentionally does no KV selection work. Its inputs, outputs and
attribute match `KVSelect`; the AICPU kernel returns success without writing
outputs. Benchmarks should use it only to measure AICPU launch/dispatch behavior.

## Layout

- `op_host/mock_kv_select_def.cpp`: OpDef used by opbuild/aclnn generation.
- `op_proto/mock_kv_select_proto.cpp`: shape and dtype inference.
- `cpukernel/impl/*`: AICPU kernel implementation and registration.
- `cpukernel/op_info_cfg/aicpu_kernel/mock_kv_select.ini`: AICPU op info.
- `CMakeLists.txt`: builds `libasn_aicpu_kernels.so` and opInfo json.
- `build.sh`: local build helper.
- `scripts/install_to_opp.sh`: copies generated AICPU artifacts into OPP.
- `tests/test_mock_kv_select_acl.py`: Python/ACL smoke test through the handwritten ACLNN wrapper.

## Build

```bash
cd op/aicpu_mock_kv_select
./build.sh --cann /usr/local/Ascend/cann-8.5.1
```

The primary outputs are under `build/output/vendors/customize_asn`:

- `op_impl/cpu/aicpu_kernel/impl/libasn_aicpu_kernels.so`
- `op_impl/cpu/config/cust_aicpu_kernel.json`
- `op_api/lib/libcust_opapi.so`
- `op_proto/lib/linux/<arch>/libcust_opsproto_mock_kv_select.so`

Install these into the active OPP with:

```bash
./scripts/install_to_opp.sh --cann /usr/local/Ascend/cann-8.5.1
```

Run the ACL smoke test from an environment with CANN variables loaded:

```bash
source /usr/local/Ascend/cann-8.5.1/opp/vendors/customize_asn/bin/set_env.bash
export LD_LIBRARY_PATH=$PWD/build/output/vendors/customize_asn/op_api/lib:$LD_LIBRARY_PATH
python3 tests/test_mock_kv_select_acl.py
```

The install script only installs the AICPU JSON/so by default. Pass
`--install-aclnn` only if you intentionally want to copy `libcust_opapi.so` into
the vendor OPP directory.
