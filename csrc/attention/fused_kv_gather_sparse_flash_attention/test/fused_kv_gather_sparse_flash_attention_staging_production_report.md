# 性能评估结果

生成时间：2026-08-30T14:26:30

主标杆为已完成一次 KV materialization 后的 `SparseFlashAttention`；
另附语义等价的 `DsaKvGather + SparseFlashAttention` 完整链路。
固定 schedule：warmup=5、active=5；
指标为 `op_statistic.csv` 中目标 OP Type 的 `Total Time(us) / 5`。
用例：`/data/wangpeng/vllm-ascend/csrc/attention/fused_kv_gather_sparse_flash_attention/test/fused_kv_gather_sparse_flash_attention_perf_cases.jsonl`；trace：`/tmp/fused-staging-profile`。所有 case 均先完成非零 Hot/Host 精度对比。

## 性能对比

| Case | Shape | DType | 融合(us) | SFA(us) | 相对SFA比值 |
| ---- | ----- | ----- | -------: | ------: | ----------: |
| 7 | BS=12, MTP=3, S=65536, miss=0.1 | bfloat16 | 240.993 | 150.267 | 0.624 |

## 全量汇总

| 指标 | 值 |
| ---- | --: |
| 用例数 | 1 |
| 平均相对SFA比值（>1 表示融合更快） | 0.624 |
| 融合算子更优 | 0 |
| SFA 单算子更优 | 1 |

### 按数据类型汇总

| DType | 用例数 | 平均相对SFA比值 | 融合更优 | SFA更优 |
| ----- | -----: | ----------------: | -------: | ------: |
| bfloat16 | 1 | 0.624 | 0 | 1 |

## 与未融合完整链路对比

| Case | Shape | DType | 融合(us) | Gather + SFA(us) | 链路加速比 |
| ---- | ----- | ----- | -------: | -----------------: | ---------: |
| 7 | BS=12, MTP=3, S=65536, miss=0.1 | bfloat16 | 240.993 | 720.426 | 2.989 |

## 简短分析

- 相对 SFA 单算子平均比值为 0.624，用于衡量 route 解析和直接 Host/Hot 装载的附加成本。
- 相对 Gather + SFA 完整链路平均加速比为 2.989，用于判断融合是否真正消除 selection-cache 往返。
- BS=12/MTP=3/S=64K 是主要生产门禁；只有完整链路加速且精度通过时才应接入默认路径。
