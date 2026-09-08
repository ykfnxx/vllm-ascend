# 性能评估结果

生成时间：2026-08-29T17:31:22

主标杆为已完成一次 KV materialization 后的 `SparseFlashAttention`；
另附语义等价的 `DsaKvGather + SparseFlashAttention` 完整链路。
固定 schedule：warmup=5、active=5；
指标为 `op_statistic.csv` 中目标 OP Type 的 `Total Time(us) / 5`。
用例：`/data/wangpeng/vllm-ascend/csrc/attention/fused_kv_gather_sparse_flash_attention/test/fused_kv_gather_sparse_flash_attention_perf_cases.jsonl`；trace：`/data/wangpeng/vllm-ascend/csrc/attention/fused_kv_gather_sparse_flash_attention/test/profiler_trace`。所有 case 均先完成非零 Hot/Host 精度对比。

## 性能对比

| Case | Shape | DType | 融合(us) | SFA(us) | 相对SFA比值 |
| ---- | ----- | ----- | -------: | ------: | ----------: |
| 0 | BS=1, MTP=0, S=4096, miss=0.1 | float16 | 118.482 | 80.950 | 0.683 |
| 1 | BS=1, MTP=0, S=4096, miss=0.1 | bfloat16 | 117.410 | 80.454 | 0.685 |
| 2 | BS=4, MTP=3, S=16384, miss=0.1 | float16 | 123.526 | 87.258 | 0.706 |
| 3 | BS=4, MTP=3, S=16384, miss=0.1 | bfloat16 | 123.502 | 86.834 | 0.703 |
| 4 | BS=12, MTP=3, S=16384, miss=0.1 | float16 | 221.204 | 152.975 | 0.692 |
| 5 | BS=12, MTP=3, S=16384, miss=0.1 | bfloat16 | 221.120 | 154.823 | 0.700 |
| 6 | BS=12, MTP=3, S=65536, miss=0.1 | float16 | 222.192 | 151.403 | 0.681 |
| 7 | BS=12, MTP=3, S=65536, miss=0.1 | bfloat16 | 221.392 | 153.983 | 0.696 |

## 全量汇总

| 指标 | 值 |
| ---- | --: |
| 用例数 | 8 |
| 平均相对SFA比值（>1 表示融合更快） | 0.693 |
| 融合算子更优 | 0 |
| SFA 单算子更优 | 8 |

### 按数据类型汇总

| DType | 用例数 | 平均相对SFA比值 | 融合更优 | SFA更优 |
| ----- | -----: | ----------------: | -------: | ------: |
| bfloat16 | 4 | 0.696 | 0 | 4 |
| float16 | 4 | 0.691 | 0 | 4 |

## 与未融合完整链路对比

| Case | Shape | DType | 融合(us) | Gather + SFA(us) | 链路加速比 |
| ---- | ----- | ----- | -------: | -----------------: | ---------: |
| 0 | BS=1, MTP=0, S=4096, miss=0.1 | float16 | 118.482 | 105.902 | 0.894 |
| 1 | BS=1, MTP=0, S=4096, miss=0.1 | bfloat16 | 117.410 | 103.742 | 0.884 |
| 2 | BS=4, MTP=3, S=16384, miss=0.1 | float16 | 123.526 | 237.173 | 1.920 |
| 3 | BS=4, MTP=3, S=16384, miss=0.1 | bfloat16 | 123.502 | 236.517 | 1.915 |
| 4 | BS=12, MTP=3, S=16384, miss=0.1 | float16 | 221.204 | 708.554 | 3.203 |
| 5 | BS=12, MTP=3, S=16384, miss=0.1 | bfloat16 | 221.120 | 704.158 | 3.185 |
| 6 | BS=12, MTP=3, S=65536, miss=0.1 | float16 | 222.192 | 713.174 | 3.210 |
| 7 | BS=12, MTP=3, S=65536, miss=0.1 | bfloat16 | 221.392 | 711.426 | 3.213 |

## 简短分析

- 相对 SFA 单算子平均比值为 0.693，用于衡量 route 解析和直接 Host/Hot 装载的附加成本。
- 相对 Gather + SFA 完整链路平均加速比为 2.303，用于判断融合是否真正消除 selection-cache 往返。
- BS=12/MTP=3/S=64K 是主要生产门禁；只有完整链路加速且精度通过时才应接入默认路径。
