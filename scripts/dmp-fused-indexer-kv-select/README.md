# LightningIndexer + IndexUpdate 融合算子

## Build

```bash
EVICT_EXTRA_SCAN_CHUNKS=8 BUILD_JOBS=16 SOC_VERSION=ascend910_9391 bash build_and_install.sh
```

The script builds and installs the CANN custom op locally, then builds the torch extension in place. It records the local OPP install path in `.lightning_indexer_decode_env`, so the test scripts can load the custom op without `pip install -e`.

`EVICT_EXTRA_SCAN_CHUNKS` is a compile-time option in `[0, 512]` and defaults to `8`. It controls how many additional unvisited chunks are scanned after enough evict candidates are found. Test data construction does not read this option.

## Benchmark / Test

Correctness tests:

```bash
python tests/test.py --device 0 --bs 24 --min-seqlen 32768 --max-seqlen 65536 --cache-size 8192 --min-miss-count 0 --max-miss-count 200
```

Performance tests:

```bash
python tests/perf.py --device 0 --bs 24 --min-seqlen 32768 --max-seqlen 65536 --cache-size 8192 --min-miss-count 0 --max-miss-count 200 --iters 20
```

Performance and Correctness scan:

```
for bs in 24; do
  for miss_range in \
    "0 0" \
    "50 200" \
    "200 500" \
    "0 2048"; do

    read -r min_miss max_miss <<< "$miss_range"
    echo "===== bs=${bs} seqlen=32768-65536 miss=${min_miss}-${max_miss} ====="

    python tests/test.py \
      --device 0 \
      --bs "$bs" \
      --min-seqlen 32768 \
      --max-seqlen 65536 \
      --cache-size 12288 \
      --min-miss-count "$min_miss" \
      --max-miss-count "$max_miss"

    python tests/perf.py \
      --device 0 \
      --bs "$bs" \
      --min-seqlen 32768 \
      --max-seqlen 65536 \
      --cache-size 12288 \
      --min-miss-count "$min_miss" \
      --max-miss-count "$max_miss" \
      --iters 100
  done
done
```
