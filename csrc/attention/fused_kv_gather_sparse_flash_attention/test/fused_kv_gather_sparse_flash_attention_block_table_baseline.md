# Block-table optimization baseline

Baseline source: `fused_kv_gather_sparse_flash_attention_optimized_report.md`,
generated after the dedicated route-tile optimization and before block-table
cache-line grouping.

| Case | Shape | DType | Fused (us) | Gather + SFA (us) |
| --- | --- | --- | ---: | ---: |
| 0 | BS=1, MTP=0, S=4096, miss=0.1 | float16 | 116.906 | 104.238 |
| 1 | BS=1, MTP=0, S=4096, miss=0.1 | bfloat16 | 117.334 | 104.170 |
| 2 | BS=4, MTP=3, S=16384, miss=0.1 | float16 | 123.106 | 238.237 |
| 3 | BS=4, MTP=3, S=16384, miss=0.1 | bfloat16 | 122.198 | 240.433 |
| 4 | BS=12, MTP=3, S=16384, miss=0.1 | float16 | 217.472 | 706.422 |
| 5 | BS=12, MTP=3, S=16384, miss=0.1 | bfloat16 | 216.884 | 706.602 |
| 6 | BS=12, MTP=3, S=65536, miss=0.1 | float16 | 218.628 | 712.882 |
| 7 | BS=12, MTP=3, S=65536, miss=0.1 | bfloat16 | 219.940 | 718.498 |

The four BS=12/MTP=3 production cases average 218.231 us.
