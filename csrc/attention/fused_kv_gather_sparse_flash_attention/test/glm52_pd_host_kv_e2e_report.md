# GLM-5.2 PD Host KV 真实端到端验收

测试日期：2026-08-31

## 环境与配置

- 容器：`va23-asu-wp`
- 硬件：16 张 A3；rank 0--7 为 Prefill TP8，rank 8--15 为 Decode TP8
- 模型：`/data/model/GLM-5.2-w4a8`
- Decode：MTP=3、DSA Sparse Host backend、LocalShm PD 传输、FULL_DECODE_ONLY Graph
- 物理层数：78；IndexShare cohort 数：21

## 真实模型验收结果

| 场景 | Prompt | Completion | 结果 | 关键证据 |
| --- | ---: | ---: | --- | --- |
| 短上下文冒烟 | 4,096 | 4 | PASS | 312 次融合 SFA；84 次 lookup/update；真实 Host KV/RoPE |
| 长上下文 | 65,536 | 4 | PASS | HTTP 200；312 次融合 SFA；84 次 lookup/update；LocalShm 使用约 51 GiB |

64K 用例使用 `max_model_len=65540`、`max_num_batched_tokens=8192` 和
`gpu_memory_utilization=0.96`。Prefill KV 容量为 71,672 token，Decode KV 容量为
924,842 token。验证器输出：

```text
PASS: {"accepted_tail_commit":312,"cohort_count":21,"completion_tokens":4,
"fused_kv_gather_sfa":312,"history_load_mock":0,"hot_cache_sfa_done":0,
"layer_count":78,"lookup_update_done":84,"target_steps":4,"tp_rank_count":8}
```

`history_load_mock=0`，Prefill/Decode 日志中的分层 SHA256 映射和约 51 GiB
LocalShm 实际占用共同证明：测试走了真实 PD KV 传输和 Host KV/RoPE，而不是
mock payload。

## 64K Decode 八 rank profiler 汇总

下表对 8 个 Decode rank 的 `op_statistic.csv` 做加权汇总。

| OP Type | 每 rank 次数 | 八 rank 加权平均耗时 |
| --- | ---: | ---: |
| `DsaKvGather` | 78 | 1,227.160 us |
| `FusedKvGatherSparseFlashAttention` | 312 | 164.239 us |
| `DsaUpdate` | 84 | 359.713 us |
| `AsuKvGather` | 312 | 25.444 us |
| `LightningIndexer` | 96 | 61.210 us |
| `DsaLookup` | 84 | 29.018 us |
| `SparseFlashAttention` | 12 | 71.569 us |

78 次 `DsaKvGather` 来自 `HostMappedDSASparseKVBackend.store_host_blocks()`：
每个物理层在 PD 接收后把完整 64K KV/RoPE 从 HBM staging 写入 swapped Host
memory 一次。它不是逐 Decode step 的稳态 gather。稳态每层、每 step 的
payload install 是 312 次 `AsuKvGather`，平均 25.444 us。

84 次 `DsaLookup/DsaUpdate` 等于 21 个 IndexShare cohort 乘 4 个 target step；
312 次融合 SFA 等于 78 个物理层乘 4 个 target step。

## BS=12、64K、MTP=3、10% miss 算子级 A/B

生产 shape 的 NPU Event 中位数，warmup=5、iterations=30：

| 路径 | 中位耗时 |
| --- | ---: |
| staging HBM -> Hot HBM install | 74.800 us |
| Host -> Hot HBM 重复读取 | 197.180 us |
| `DsaKvGather + SparseFlashAttention` | 约 507.8 us |
| `FusedKvGatherSparseFlashAttention` | 247.140 us |
| AIV `DsaUpdate + fused SFA` 串行 | 514.580 us |
| AIV `DsaUpdate` 与 fused SFA 辅助流并发 | 732.730 us |
| `DsaUpdateAicpu + fused SFA` 串行 | 580.120 us |
| `DsaUpdateAicpu` 与 fused SFA 辅助流并发 | 350.580 us |

AIV Update 与融合 SFA 会争用 Vector Core，辅助流反而退化 42.4%。真正的
AICPU Update 不占用 AIV，BS=12 时可把该二算子链从 580.120 us 降到
350.580 us。BS=1 时，AICPU Update 单算子为 106.190 us，AIV Update 为
254.360 us；但事件/侧流开销使 AICPU 串行链 198.750 us 优于辅助流链
234.760 us。因此调度策略应按 batch 选择，不能无条件使用辅助流。

当前真实 GLM 路径仍是独立 `LightningIndexer + DsaLookup`，没有生成
`DsaUpdateAicpu` 所需的 compact miss journal/protected bitmap，所以 64K 模型
profiler 中仍显示 AIV `DsaUpdate`。AICPU 路径已完成打包、符号导出和真实
correctness，但尚不能在不启用较慢的 FusedLightningIndexerLookup 时接入模型。

## 精度与边界

- 独立精度矩阵：FP16/BF16、BS 1--49、MTP 0/3、4K--128K、miss 0%--100%，
  共 30 个用例全部 PASS，融合输出及 staging payload 最大误差均为 0。
- 真实 GLM 64K 验收为 BS=1。BS=12、64K 已完成算子级 profile，但 12 个并发
  64K Prefill 至少需要 786,432 token 的 Prefill KV 容量，超过当前 71,672 token；
  真正并发验收需要 Prefill 侧按块即时卸载/回收，而不是只提高内存利用率。
- 64K 请求与验证均已 PASS；随后在线 profiler daemon 与离线解析同时运行导致
  worker 退出。该问题属于 profiler 生命周期，后续应在停止服务后再离线解析，
  不能把当前结果表述为长时间在线稳定性验收。
