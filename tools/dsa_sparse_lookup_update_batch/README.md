# dsa_sparse_lookup_update_batch tools

Build and install the independent Ascend 950 batch operator:

```bash
bash tools/dsa_sparse_lookup_update_batch/build_and_install.sh --fresh
```

The Python extension must also be rebuilt once after pulling this branch so
that `torch.ops._C_ascend.dsa_sparse_lookup_update_batch` is registered. The
single-op script above builds and installs only the CANN custom-op package.

Run the packed-query correctness test:

```bash
python3 tools/dsa_sparse_lookup_update_batch/test_correctness.py \
  --device npu:0 \
  --requests 2 \
  --queries-per-request 4
```

Measure operator-only event latency with 8K resident slots per request:

```bash
python3 tools/dsa_sparse_lookup_update_batch/benchmark_operator.py \
  --device npu:0 \
  --concurrency 8 \
  --queries-per-request 4 \
  --miss-rate 10
```

The timed interval starts after state restoration and contains only one
`dsa_sparse_lookup_update_batch` launch. `--miss-rate` is applied per packed
TopK row; the report separately records successful misses and fallback count.

Collect an operator trace with the same workload shape:

```bash
python3 tools/dsa_sparse_lookup_update_batch/profile_operator.py \
  --device npu:0 \
  --concurrency 8 \
  --queries-per-request 4 \
  --miss-rate 10 \
  --output-dir /tmp/dsa-sparse-batch-profile
```
