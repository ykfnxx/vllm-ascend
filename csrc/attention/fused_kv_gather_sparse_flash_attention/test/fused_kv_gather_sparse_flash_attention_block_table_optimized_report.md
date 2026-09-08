# 性能评估结果

生成时间：2026-08-30T03:08:45

主标杆为已完成一次 KV materialization 后的 `SparseFlashAttention`；
另附语义等价的 `DsaKvGather + SparseFlashAttention` 完整链路。
固定 schedule：warmup=5、active=5；
指标为 `op_statistic.csv` 中目标 OP Type 的 `Total Time(us) / 5`。
用例：`/data/wangpeng/vllm-ascend/csrc/attention/fused_kv_gather_sparse_flash_attention/test/fused_kv_gather_sparse_flash_attention_perf_cases.jsonl`；trace：`/tmp/fused-kv-block-table-cacheline`。所有 case 均先完成非零 Hot/Host 精度对比。

## 性能对比

| Case | Shape | DType | 融合(us) | SFA(us) | 相对SFA比值 |
| ---- | ----- | ----- | -------: | ------: | ----------: |
| 0 | BS=1, MTP=0, S=4096, miss=0.1 | float16 | 127.207 | 81.482 | 0.641 |
| 1 | BS=1, MTP=0, S=4096, miss=0.1 | bfloat16 | 127.519 | 82.182 | 0.644 |
| 2 | BS=4, MTP=3, S=16384, miss=0.1 | float16 | 134.111 | 86.414 | 0.644 |
| 3 | BS=4, MTP=3, S=16384, miss=0.1 | bfloat16 | 138.715 | 86.906 | 0.627 |
| 4 | BS=12, MTP=3, S=16384, miss=0.1 | float16 | 256.297 | 161.671 | 0.631 |
| 5 | BS=12, MTP=3, S=16384, miss=0.1 | bfloat16 | 254.097 | 165.435 | 0.651 |
| 6 | BS=12, MTP=3, S=65536, miss=0.1 | float16 | 265.205 | 160.079 | 0.604 |
| 7 | BS=12, MTP=3, S=65536, miss=0.1 | bfloat16 | 263.241 | 156.619 | 0.595 |

## 全量汇总

| 指标 | 值 |
| ---- | --: |
| 用例数 | 8 |
| 平均相对SFA比值（>1 表示融合更快） | 0.630 |
| 融合算子更优 | 0 |
| SFA 单算子更优 | 8 |

### 按数据类型汇总

| DType | 用例数 | 平均相对SFA比值 | 融合更优 | SFA更优 |
| ----- | -----: | ----------------: | -------: | ------: |
| bfloat16 | 4 | 0.629 | 0 | 4 |
| float16 | 4 | 0.630 | 0 | 4 |

## 与未融合完整链路对比

| Case | Shape | DType | 融合(us) | Gather + SFA(us) | 链路加速比 |
| ---- | ----- | ----- | -------: | -----------------: | ---------: |
| 0 | BS=1, MTP=0, S=4096, miss=0.1 | float16 | 127.207 | 104.582 | 0.822 |
| 1 | BS=1, MTP=0, S=4096, miss=0.1 | bfloat16 | 127.519 | 104.058 | 0.816 |
| 2 | BS=4, MTP=3, S=16384, miss=0.1 | float16 | 134.111 | 238.113 | 1.775 |
| 3 | BS=4, MTP=3, S=16384, miss=0.1 | bfloat16 | 138.715 | 236.501 | 1.705 |
| 4 | BS=12, MTP=3, S=16384, miss=0.1 | float16 | 256.297 | 712.786 | 2.781 |
| 5 | BS=12, MTP=3, S=16384, miss=0.1 | bfloat16 | 254.097 | 709.214 | 2.791 |
| 6 | BS=12, MTP=3, S=65536, miss=0.1 | float16 | 265.205 | 719.918 | 2.715 |
| 7 | BS=12, MTP=3, S=65536, miss=0.1 | bfloat16 | 263.241 | 720.338 | 2.736 |

## 简短分析

- 相对 SFA 单算子平均比值为 0.630，用于衡量 route 解析和直接 Host/Hot 装载的附加成本。
- 相对 Gather + SFA 完整链路平均加速比为 2.018，用于判断融合是否真正消除 selection-cache 往返。
- BS=12/MTP=3/S=64K 是主要生产门禁；只有完整链路加速且精度通过时才应接入默认路径。
