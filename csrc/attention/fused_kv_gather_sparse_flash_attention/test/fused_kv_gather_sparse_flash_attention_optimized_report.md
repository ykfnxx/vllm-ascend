# 性能评估结果

生成时间：2026-08-30T03:18:09

主标杆为已完成一次 KV materialization 后的 `SparseFlashAttention`；
另附语义等价的 `DsaKvGather + SparseFlashAttention` 完整链路。
固定 schedule：warmup=5、active=5；
指标为 `op_statistic.csv` 中目标 OP Type 的 `Total Time(us) / 5`。
用例：`/data/wangpeng/vllm-ascend/csrc/attention/fused_kv_gather_sparse_flash_attention/test/fused_kv_gather_sparse_flash_attention_perf_cases.jsonl`；trace：`/tmp/fused-kv-dcache-final`。所有 case 均先完成非零 Hot/Host 精度对比。

## 性能对比

| Case | Shape | DType | 融合(us) | SFA(us) | 相对SFA比值 |
| ---- | ----- | ----- | -------: | ------: | ----------: |
| 0 | BS=1, MTP=0, S=4096, miss=0.1 | float16 | 111.078 | 80.142 | 0.721 |
| 1 | BS=1, MTP=0, S=4096, miss=0.1 | bfloat16 | 110.422 | 82.146 | 0.744 |
| 2 | BS=4, MTP=3, S=16384, miss=0.1 | float16 | 116.650 | 85.414 | 0.732 |
| 3 | BS=4, MTP=3, S=16384, miss=0.1 | bfloat16 | 117.282 | 86.994 | 0.742 |
| 4 | BS=12, MTP=3, S=16384, miss=0.1 | float16 | 207.820 | 155.695 | 0.749 |
| 5 | BS=12, MTP=3, S=16384, miss=0.1 | bfloat16 | 207.556 | 156.911 | 0.756 |
| 6 | BS=12, MTP=3, S=65536, miss=0.1 | float16 | 214.136 | 154.867 | 0.723 |
| 7 | BS=12, MTP=3, S=65536, miss=0.1 | bfloat16 | 213.840 | 155.463 | 0.727 |

## 全量汇总

| 指标 | 值 |
| ---- | --: |
| 用例数 | 8 |
| 平均相对SFA比值（>1 表示融合更快） | 0.737 |
| 融合算子更优 | 0 |
| SFA 单算子更优 | 8 |

### 按数据类型汇总

| DType | 用例数 | 平均相对SFA比值 | 融合更优 | SFA更优 |
| ----- | -----: | ----------------: | -------: | ------: |
| bfloat16 | 4 | 0.742 | 0 | 4 |
| float16 | 4 | 0.732 | 0 | 4 |

## 与未融合完整链路对比

| Case | Shape | DType | 融合(us) | Gather + SFA(us) | 链路加速比 |
| ---- | ----- | ----- | -------: | -----------------: | ---------: |
| 0 | BS=1, MTP=0, S=4096, miss=0.1 | float16 | 111.078 | 102.482 | 0.923 |
| 1 | BS=1, MTP=0, S=4096, miss=0.1 | bfloat16 | 110.422 | 102.830 | 0.931 |
| 2 | BS=4, MTP=3, S=16384, miss=0.1 | float16 | 116.650 | 236.921 | 2.031 |
| 3 | BS=4, MTP=3, S=16384, miss=0.1 | bfloat16 | 117.282 | 236.909 | 2.020 |
| 4 | BS=12, MTP=3, S=16384, miss=0.1 | float16 | 207.820 | 711.190 | 3.422 |
| 5 | BS=12, MTP=3, S=16384, miss=0.1 | bfloat16 | 207.556 | 703.694 | 3.390 |
| 6 | BS=12, MTP=3, S=65536, miss=0.1 | float16 | 214.136 | 720.426 | 3.364 |
| 7 | BS=12, MTP=3, S=65536, miss=0.1 | bfloat16 | 213.840 | 715.822 | 3.347 |

## 简短分析

- 相对 SFA 单算子平均比值为 0.737，用于衡量 route 解析和直接 Host/Hot 装载的附加成本。
- 相对 Gather + SFA 完整链路平均加速比为 2.429，用于判断融合是否真正消除 selection-cache 往返。
- BS=12/MTP=3/S=64K 是主要生产门禁；只有完整链路加速且精度通过时才应接入默认路径。
