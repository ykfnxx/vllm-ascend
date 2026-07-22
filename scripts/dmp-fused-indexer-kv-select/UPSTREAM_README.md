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
python tests/test.py --device 0 --bs 24 --min-seqlen 32768 --max-seqlen 65536 --cache-size 10240 --min-miss-count 0 --max-miss-count 500
```

Performance tests:

```bash
python tests/perf.py --device 0 --bs 24 --min-seqlen 32768 --max-seqlen 65536 --cache-size 10240 --min-miss-count 0 --max-miss-count 500 --iters 20
```

Performance and Correctness scan:

```
for bs in 24 48; do
  for seqlen in 65536 131072; do
    for miss_range in \
    "0 0" \
    "0 300" \
    "0 2048"; do

    read -r min_miss max_miss <<< "$miss_range"
    echo "===== bs=${bs} seqlen=${seqlen} miss=${min_miss}-${max_miss} ====="

    python tests/test.py \
      --device 0 \
      --bs "$bs" \
      --min-seqlen ${seqlen} \
      --max-seqlen ${seqlen} \
      --cache-size 10240 \
      --min-miss-count "$min_miss" \
      --max-miss-count "$max_miss"

    python tests/perf.py \
      --device 0 \
      --bs "$bs" \
      --min-seqlen ${seqlen} \
      --max-seqlen ${seqlen} \
      --cache-size 10240 \
      --min-miss-count "$min_miss" \
      --max-miss-count "$max_miss" \
      --iters 100

    python test_compare_li/perf_indexer.py \
      --device 0 \
      --bs "$bs" \
      --min-seqlen ${seqlen} \
      --max-seqlen ${seqlen} \
      --cache-size 10240 \
      --min-miss-count "$min_miss" \
      --max-miss-count "$max_miss" \
      --iters 100
    done
  done
done
```

以上命令测出来的时延结果汇总如下两张表：

### 如果编译时采用 EVICT_EXTRA_SCAN_CHUNKS=2

| Batch size | Sequence length | Miss count 范围 | LightningIndexer(vLLM) | LightningIndexerDecode | LightningIndexerDecodeUpdate | Update相对原版 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 24 | 65536 | 0 | 440 μs | 416 μs | 436 μs | -4 μs |
| 24 | 65536 | 0..300 | 442 μs | 413 μs | 462 μs | 20 μs |
| 24 | 65536 | 0..2048 | 442 μs | 417 μs | 613 μs | 171 μs |
| 24 | 131072 | 0 | 812 μs | 759 μs | 803 μs | -9 μs |
| 24 | 131072 | 0..300 | 800 μs | 760 μs | 832 μs | 32 μs |
| 24 | 131072 | 0..2048 | 804 μs | 761 μs | 1003 μs | 199 μs |
| 48 | 65536 | 0 | 836 μs | 787 μs | 828 μs | -8 μs |
| 48 | 65536 | 0..300 | 838 μs | 789 μs | 874 μs | 36 μs |
| 48 | 65536 | 0..2048 | 837 μs | 782 μs | 1133 μs | 296 μs |
| 48 | 131072 | 0 | 1571 μs | 1486 μs | 1551 μs | -20 μs |
| 48 | 131072 | 0..300 | 1571 μs | 1483 μs | 1603 μs | 32 μs |
| 48 | 131072 | 0..2048 | 1565 μs | 1492 μs | 1933 μs | 368 μs |

### 如果编译时采用 EVICT_EXTRA_SCAN_CHUNKS=8

| Batch size | Sequence length | Miss count 范围 | LightningIndexer(vLLM) | LightningIndexerDecode | LightningIndexerDecodeUpdate | Update相对原版 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 24 | 65536 | 0 | 439 μs | 416 μs | 429 μs | -10 μs |
| 24 | 65536 | 0..300 | 430 μs | 409 μs | 471 μs | 41 μs |
| 24 | 65536 | 0..2048 | 435 μs | 409 μs | 601 μs | 166 μs |
| 24 | 131072 | 0 | 802 μs | 764 μs | 789 μs | -13 μs |
| 24 | 131072 | 0..300 | 799 μs | 754 μs | 825 μs | 26 μs |
| 24 | 131072 | 0..2048 | 794 μs | 759 μs | 1002 μs | 208 μs |
| 48 | 65536 | 0 | 840 μs | 785 μs | 819 μs | -21 μs |
| 48 | 65536 | 0..300 | 822 μs | 789 μs | 883 μs | 61 μs |
| 48 | 65536 | 0..2048 | 825 μs | 785 μs | 1137 μs | 312 μs |
| 48 | 131072 | 0 | 1562 μs | 1484 μs | 1544 μs | -18 μs |
| 48 | 131072 | 0..300 | 1556 μs | 1481 μs | 1608 μs | 52 μs |
| 48 | 131072 | 0..2048 | 1554 μs | 1471 μs | 1970 μs | 416 μs |

其中 `Update相对原版 = LightningIndexerDecodeUpdate - LightningIndexer(VLLM)`
