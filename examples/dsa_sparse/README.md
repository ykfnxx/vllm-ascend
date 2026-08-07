# DSA Sparse 测试、验证与调试脚本

本目录中的脚本用于验证 GLM-5 DSA Sparse 框架、ASU HBM 索引算子、
mock KV backend、并发请求和 Ascend profiler。除特别说明外，命令都应从
`vllm-ascend` 仓库根目录执行。

## 脚本总览

| 脚本 | 用途 | 是否需要 NPU | 是否需要已启动的服务 |
| --- | --- | --- | --- |
| [check_asu_hbm_index_ops.py](../../check_asu_hbm_index_ops.py) | 对照 CPU 参考实现检查 lookup 和 maintain 功能 | 是 | 否 |
| [smoke_asu_hbm_index_npugraph.py](smoke_asu_hbm_index_npugraph.py) | 检查两个算子能否进入 `npugraph_ex` 并回放 | 是 | 否 |
| [benchmark_asu_hbm_index_ops.py](benchmark_asu_hbm_index_ops.py) | 测量单算子在不同 batch 下的 NPU 执行时间 | 是 | 否 |
| [compare_asu_hbm_index_maintain_modes.py](compare_asu_hbm_index_maintain_modes.py) | 对比多层双 microbatch Maintain 序列的 eager 与 ACL NPU Graph replay 时延，并可分别采集 trace | 是 | 否 |
| [profile_asu_hbm_index_ops.py](profile_asu_hbm_index_ops.py) | 单独采集一个 ASU 算子并直接解析 profiler 数据 | 是 | 否 |
| [serve_glm5_dsa_sparse.sh](serve_glm5_dsa_sparse.sh) | 以 mock KV backend 启动 DSA Sparse 服务 | 是 | 脚本负责启动 |
| [diagnose_glm5_dsa_sparse_env.py](diagnose_glm5_dsa_sparse_env.py) | 静态检查运行中服务、安装包、模型和日志 | 否 | 是 |
| [verify_glm5_dsa_sparse_ops.py](verify_glm5_dsa_sparse_ops.py) | 发送长请求并验证两个算子的调用日志 | 服务端需要 | 是 |
| [test_glm5_dsa_sparse_batches.py](test_glm5_dsa_sparse_batches.py) | 自动启动服务并检查多个并发 batch | 是 | 脚本负责启停 |
| [profile_glm5_dsa_sparse.py](profile_glm5_dsa_sparse.py) | 向现有服务发送可控并发请求并控制采集窗口 | 服务端需要 | 是 |
| [run_glm5_dsa_sparse_profiles.sh](run_glm5_dsa_sparse_profiles.sh) | 按 batch 独立启停服务、采集并归档 profile | 是 | 脚本负责启停 |
| [parse_glm5_dsa_profile.py](parse_glm5_dsa_profile.py) | 离线解析原始 Ascend profiler 数据 | 不执行模型 | 否 |

## 前置条件

1. 当前分支的 `vllm-ascend` 已经安装到测试环境，并包含编译后的 custom
   ops、AICPU OPP 和 `vllm.general_plugins` 入口。
2. 使用与当前分支匹配的 vLLM Ascend、PyTorch NPU 和 CANN 环境。
3. 服务级测试使用满足以下配置的 GLM-5 模型：
   `architectures` 包含 `GlmMoeDsaForCausalLM`，`model_type` 为
   `glm_moe_dsa`，`index_topk` 为 2048。
4. DSA Sparse 当前阈值为 `8192 + 2048 + 128 = 10368` token。服务级测试的
   prompt 必须超过该值，默认使用至少 10600 token。
5. 请求至少生成 2 个 completion token，才能确认进入 decode 计算。

`ASCEND_CUSTOM_OPP_PATH` 在启动服务前可以为空。服务会在
`NPUPlatform.import_kernels` 中追加已安装包内的 AIV 和 AICPU OPP 路径。
单算子脚本会在导入 `torch_npu` 之前主动设置该变量。

## 推荐验证顺序

建议按照以下顺序缩小问题范围：

1. 检查 AICPU 安装产物和动态符号。
2. 直接执行两个算子并与 CPU 参考实现比较。
3. 检查两个算子能否被 `npugraph_ex` 捕获和回放。
4. 启动 DSA Sparse 服务并运行环境诊断。
5. 使用长请求验证完整调用链。
6. 运行多 batch 稳定性测试或单算子性能测试。
7. 最后进行服务级 profiling 和离线解析。

## 1. AICPU 安装产物诊断

以下命令不执行算子，只检查 AICPU JSON、共享库、`RunCpuKernel` 动态符号和
内嵌算子注册名：

```bash
python3 check_asu_hbm_index_ops.py --diagnose-aicpu
```

成功时最后输出：

```text
[PASS] packaged AICPU metadata and binary entry are consistent
```

该模式依赖仓库中的
`vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_impl/aicpu_transformer`
构建产物，并使用 `readelf` 检查动态符号。

## 2. 算子功能正确性

```bash
python3 check_asu_hbm_index_ops.py \
  --device-id 0 \
  --batch-size 8 \
  --seed 20260714
```

`--batch-size` 同时控制：

- `query_index` 和 `req_pool_entries` 的请求数；
- lookup 和 maintain 的 `req_num`；
- CPU 参考实现处理的请求数；
- 使用到的独立索引 pool entry 数量。

每个请求固定查询 2048 个 token，其中 1024 个 hit、1024 个 miss。脚本逐项
比较 lookup 的 slot/miss 输出、`free_head`、双向索引，以及 maintain 后的
双向索引、`free_slots` 和 `free_head`。

全部通过时最后输出：

```text
ASU HBM index custom-op check passed: ...
```

不同 batch 需要分别运行：

```bash
for batch_size in 1 2 4 8 16; do
  python3 check_asu_hbm_index_ops.py --batch-size "${batch_size}"
done
```

## 3. `npugraph_ex` 入图检查

```bash
python3 examples/dsa_sparse/smoke_asu_hbm_index_npugraph.py \
  --device-id 0
```

脚本执行以下检查：

- 两个 PyTorch custom op 均已注册；
- 两个 op 均存在 `PrivateUse1` 和 `Meta` 实现；
- lookup 和 maintain 可以放在同一个 `torch.nn.Module.forward` 中；
- `torch.compile(..., backend="npugraph_ex", fullgraph=True)` 可以完成首次捕获；
- 第二次 forward 可以完成图回放；
- 捕获和回放后的 slot、miss、双向索引及 `free_head` 正确。

成功输出：

```text
[PASS] npugraph_ex capture and first forward
[PASS] npugraph_ex replay
[PASS] ASU HBM lookup and AICPU maintain smoke test
```

该脚本证明的是算子可以独立入图，不代表 vLLM 服务一定已经走到 DSA Sparse
调用位置。服务集成路径需要继续运行第 5 节的验证脚本。

## 4. 单算子性能测试

lookup 示例：

```bash
python3 examples/dsa_sparse/benchmark_asu_hbm_index_ops.py \
  --op lookup \
  --batch-sizes 1 2 4 8 16 \
  --miss-count 1024 \
  --warmup-iterations 10 \
  --iterations 100 \
  --output-json /data/dsa-benchmark/lookup.json
```

maintain 示例：

```bash
python3 examples/dsa_sparse/benchmark_asu_hbm_index_ops.py \
  --op maintain \
  --batch-sizes 1 2 4 8 16 \
  --miss-count 1024 \
  --fresh-maintain-tensors \
  --warmup-iterations 10 \
  --iterations 100 \
  --output-json /data/dsa-benchmark/maintain.json
```

使用 `--op all` 时会分别测量 lookup 和 maintain，不会把两个算子合并计时。

测试语义：

- batch size 直接作为算子的 `req_num`；
- 每个 batch 行使用独立的 token-to-slot 索引；
- 每个请求固定输入 2048 个 query；
- `--miss-count` 控制 lookup 的 miss 数和 maintain 的淘汰数，范围为
  0 到 2048；
- `--fresh-maintain-tensors` 为每次 maintain warmup 和正式计时预分配并保留
  一套独立的算子输入 tensor，用于排除连续调用复用相同物理地址的影响；
  tensor 分配和初始化不计入 NPU Event 时延，但显存占用会随
  `warmup-iterations + iterations` 线性增长；
- 每次 warmup 和采样前恢复同一份索引状态；启用
  `--fresh-maintain-tensors` 后，恢复目标改为每轮地址不同的预分配状态；
- 状态恢复或独立 tensor 的预分配及其同步位于 NPU Event 计时区间之外；
- 每个 case 计时前都会先执行一次功能检查。

终端输出 `mean`、`median`、`p95`、`min` 和 `requests/s`。JSON 还包含
`max`、`query_items/s`、完整配置和每次迭代的原始毫秒采样。

当 `--miss-count 0` 时 maintain 会直接处理 `free_head == 0` 的场景，接近
空操作；测试实际淘汰开销时应使用非零值。

### Maintain eager 与 ACL NPU Graph 对比

以下脚本捕获一个与 DMP Scheme 4 类似的 Maintain 序列：默认使用
4 层、每层 2 个 microbatch，共 8 个独立 workspace，并在同一个 stream
上顺序调用。它不包含 Lookup、KVGather 或其他模型算子。eager 和 graph
模式使用相同算子、调用顺序、Tensor 地址、stream、reset、seed 序列、
warmup 和 NPU Event 计时边界，唯一预期差异是直接调用完整序列或
`torch.npu.NPUGraph.replay()`：

```bash
python3 examples/dsa_sparse/compare_asu_hbm_index_maintain_modes.py \
  --batch-size 32 \
  --num-layers 4 \
  --microbatches-per-layer 2 \
  --miss-count 300 \
  --skip-check \
  --warmup-iterations 10 \
  --iterations 100 \
  --profile-output-dir /data/asu-profiles/maintain-eager-vs-graph \
  --output-json /data/dsa-benchmark/maintain-eager-vs-graph.json
```

每个 sample 的 NPU Event 覆盖完整 `层数 × microbatch 数` 调用序列，reset
全部 workspace 后只同步一次，并且 reset 和同步都位于计时区间之外。终端和
JSON 同时报告完整序列时延和 `per_maintain_mean`；后者是序列平均时延除以
调用数，不是逐算子 Event 测量。复现具体 DMP 整网时，应把
`--num-layers` 设置为 `RUN_INFO.txt` 中的 `reduced_layers`，并保持
`--microbatches-per-layer 2`。独立 workspace 的 NPU 显存占用随
`num-layers × microbatches-per-layer` 线性增长。

合成固定 workload 的 Maintain 不维护动态 `free_head` 语义，因此需要
`--skip-check`。不传 `--profile-output-dir` 时只执行 NPU Event benchmark；
传入后会在 `eager/` 和 `graph/` 下分别生成 Level2 trace，以便直接比较
AICPU task duration。该脚本使用当前分支的 `_C_ascend` 算子和 128K
workspace，不代表 DMP 独立扩展的 144K 算子结果。

### 单算子 trace 采集及直接解析

每次运行只允许选择一个算子。lookup 示例：

```bash
python3 examples/dsa_sparse/profile_asu_hbm_index_ops.py \
  --op lookup \
  --batch-size 8 \
  --output-dir /data/asu-profiles/lookup-bs8
```

maintain 示例：

```bash
python3 examples/dsa_sparse/profile_asu_hbm_index_ops.py \
  --op maintain \
  --batch-size 8 \
  --output-dir /data/asu-profiles/maintain-bs8
```

脚本默认先执行 10 次未采集 warmup，再连续采集 20 次目标算子调用。默认使用
steady-state 模式，采集窗口内不恢复状态、不执行逐轮同步，也不执行 CPU/NPU
数据比较。

如需复现 benchmark 的 reset-state 方法，增加：

```bash
python3 examples/dsa_sparse/profile_asu_hbm_index_ops.py \
  --op maintain \
  --batch-size 8 \
  --reset-state \
  --output-dir /data/asu-profiles/maintain-bs8-reset
```

`--reset-state` 会保留一份 NPU baseline；每次 warmup 和每次正式采集前，将
`index`、`slot_to_index`、`free_slots`、`free_head` 恢复到 baseline 并同步，
目标算子执行后再次同步。状态恢复不纳入目标算子本身，但 `copy_` 和同步事件会
出现在 profiler trace 中，应按 `asu_*` 或 `RunCpuKernel` 名称筛选目标耗时。

当前算子固定为每请求 2048 个 query、300 次 update 或 eviction，profiling
脚本不提供 `--miss-count`。

lookup 使用 `ProfilerLevel.Level1` 和 `AiCMetrics.PipeUtilization`；maintain
使用 `ProfilerLevel.Level2` 以保留 AI CPU 细节。两者都关闭 stack、module、
memory 和 op-args 采集。

profiler 停止后，脚本通过 `tensorboard_trace_handler` 同步解析原始数据，默认
导出 MindStudio Insight DB。也可以通过 `--export-type text` 导出文本结果。
输出目录必须为空，结果结构为：

```text
<output-dir>/
  raw/
    <trace>_ascend_pt/
      FRAMEWORK/
      PROF_*/
      ASCEND_PROFILER_OUTPUT/
  parsed/
    profiler_info*.json
    profiler_metadata.json
    ASCEND_PROFILER_OUTPUT/
      *.db
  manifest.json
```

解析失败或没有生成预期的 DB/Text 文件时，脚本返回非零状态。原始 profile
默认保留，可以继续交给 `parse_glm5_dsa_profile.py` 重新解析。

## 5. 启动服务并验证调用链

### 启动 mock backend 服务

```bash
set -o pipefail
./examples/dsa_sparse/serve_glm5_dsa_sparse.sh \
  /models/GLM-5.1-W4A8 \
  --num-gpu-blocks-override 768 \
  2>&1 | tee /tmp/glm5-dsa.log
```

脚本默认配置包括：

- `tensor_parallel_size=16`、`data_parallel_size=1`；
- `max_num_seqs=8`、`max_model_len=131072`；
- chunked prefill、prefix caching 和 eager mode；
- `block_size=128`；
- `dsa_sparse_config.enabled=true`；
- `dsa_sparse_config.kv_backend=mock`；
- 关闭 async scheduling 和 FlashComm 环境开关。

模型路径之后的参数会继续传给 `vllm serve`，可用于覆盖端口、TP、最大
并发数和 block 数等默认值。

### 使用 KVIO backend

KVIO backend 需要 worker 环境能够导入 `rdma_kv_ops`。将 DSA 配置中的
backend 和 KVIO 标识改为：

```json
{
  "dsa_sparse_config": {
    "enabled": true,
    "kv_backend": "kvio",
    "kvio_model_id": 0,
    "kvio_pd_flag": 0
  }
}
```

KV cache 分配完成后，框架会把每个本地 MLA 层的 nope/rope tensor 地址和
字节长度一次性传给 `aiv_init`，并且该初始化发生在首次模型编译或图捕获
之前。完整 block 和 lookup miss 都通过 tensor-native 的
`torch.ops._C_ascend.npu_get_put_batch` 提交，并使用相同的 `task_id` 与
`io_nums` 调用 `torch.ops._C_ascend.npu_send_wait`。PUT 使用 opcode `0x05`，
GET 使用 opcode `0x06`。每个 forward 的 block 元数据只转换成一次 NPU
`int64` tensor；GET 的 cache/storage 偏移也直接在 NPU 上生成，不再经过
`.cpu().tolist()` 或 Python descriptor loop。当前 KVIO 接口没有远端删除
API，请求结束只释放本地 DSA 请求状态。KVIO 的 `request_ids` 来自稳定的
vLLM 请求 ID，而不是会随 batch 重排或抢占而变化的 batch/pool 下标；服务
调用方应避免在同一 KVIO namespace 中复用已结束请求的 ID。

### 静态诊断运行中的服务

在服务启动后执行：

```bash
python3 examples/dsa_sparse/diagnose_glm5_dsa_sparse_env.py \
  --server-log /tmp/glm5-dsa.log
```

存在多个 `vllm serve` 进程时显式指定 PID：

```bash
python3 examples/dsa_sparse/diagnose_glm5_dsa_sparse_env.py \
  --pid 12345 \
  --model-path /models/GLM-5.1-W4A8 \
  --server-log /tmp/glm5-dsa.log
```

该脚本不会导入 torch，也不会初始化 NPU 或执行算子。它检查：

- 服务启动参数和 DSA Sparse additional config；
- 服务进程实际使用的 Python、vLLM 和 vLLM Ascend 安装位置；
- `vllm.general_plugins` 中的 DSA general plugin；
- 已安装源码是否包含当前 DSA 集成；
- custom OPP、AICPU `.so` 和 PyTorch binding；
- 模型 architecture、model type、`index_topk` 和上下文上限；
- 服务日志中的 framework、sparse threshold 和算子调用标记；
- 常见 AICPU、shared-memory broadcast 和 poller 错误。

`[WARN]` 不会导致非零退出码；存在 `[FAIL]` 时退出码为 1。

### 发送长请求验证算子调用

必须使用刚启动的服务和全新的日志文件：

```bash
python3 examples/dsa_sparse/verify_glm5_dsa_sparse_ops.py \
  --base-url http://127.0.0.1:8077 \
  --model glm-5 \
  --server-log /tmp/glm5-dsa.log \
  --prompt-tokens 10600 \
  --max-tokens 4
```

脚本通过 `/tokenize` 构造足够长的 prompt，再调用 `/v1/completions`，请求中
固定使用 `temperature=0`、`ignore_eos=true` 和非流式输出。通过条件为：

- 实际 prompt token 数超过 10368；
- completion 至少生成 2 个 token；
- 请求开始后的新增日志中同时出现 lookup 和 maintain 的调用及完成标记。

四个关键标记为：

```text
DSA sparse invoking asu_hbm_index_lookup
DSA sparse completed asu_hbm_index_lookup
DSA sparse invoking asu_hbm_index_maintain_aicpu
DSA sparse completed asu_hbm_index_maintain_aicpu
```

这些标记每个进程只打印一次。如果日志中已经存在这些标记，脚本会拒绝继续，
此时需要重启服务并使用新的日志文件。

## 6. 多 batch 稳定性测试

该脚本会自动启动一个 mock KV backend 服务、预热共享长 prefix、发送多个
同步并发请求、验证日志，然后停止服务：

```bash
python3 examples/dsa_sparse/test_glm5_dsa_sparse_batches.py \
  --model-path /models/GLM-5.1-W4A8 \
  --batch-sizes 1 2 4 8 \
  --rounds 2 \
  --num-gpu-blocks-override 768 \
  --output-dir /data/dsa-batch-tests/run-01
```

附加 vLLM 参数放在 `--` 之后：

```bash
python3 examples/dsa_sparse/test_glm5_dsa_sparse_batches.py \
  --model-path /models/GLM-5.1-W4A8 \
  --batch-sizes 1 2 4 \
  --num-gpu-blocks-override 512 \
  --output-dir /data/dsa-batch-tests/run-02 \
  -- \
  --gpu-memory-utilization 0.90
```

脚本将 `max_num_seqs` 和 `dsa_sparse_config.max_active_reqs` 设置为最大 batch。
每个请求需要 81 个 NPU block，另预留 1 个 null block，因此显式设置时必须
满足：

```text
num_gpu_blocks_override >= max_batch_size * 81 + 1
```

脚本会检查该约束，并给出向上取整到 128 的推荐值。未传入
`--num-gpu-blocks-override` 时由 vLLM 自动确定 block 数。

输出目录包含：

```text
server.log
serve-command.txt
results.json
```

除 HTTP 请求成功外，脚本还要求日志中存在 mock backend 的 put/load 标记和
lookup/maintain 的调用及完成标记。

## 7. 服务级 profiling

### 只运行 profile 客户端

服务必须使用 `--profiler-config` 启动。例如 profiler 目录配置为：

```bash
PROFILER_CONFIG=$(python3 -c \
  'import json; print(json.dumps({"profiler":"torch","torch_profiler_dir":"/data/dsa-raw"}))')

./examples/dsa_sparse/serve_glm5_dsa_sparse.sh \
  /models/GLM-5.1-W4A8 \
  --profiler-config "${PROFILER_CONFIG}" \
  --num-gpu-blocks-override 768 \
  2>&1 | tee /tmp/glm5-dsa-profile.log
```

然后运行客户端：

```bash
python3 examples/dsa_sparse/profile_glm5_dsa_sparse.py \
  --base-url http://127.0.0.1:8077 \
  --model glm-5 \
  --batch-sizes 1 2 4 8 \
  --prompt-tokens 10600 \
  --max-tokens 32 \
  --warmup-rounds 1 \
  --rounds 1 \
  --profile \
  --server-log /tmp/glm5-dsa-profile.log \
  --output-json /data/dsa-client-metrics.json
```

`--profile` 使客户端在每个 batch 的 measured wave 前后调用服务的
`/start_profile` 和 `/stop_profile`。不传该参数时只发送请求并统计客户端指标，
不会控制 profiler。

`--output-json` 只保存请求吞吐、output token 吞吐、TTFT、latency、请求参数和
日志标记状态。原始 NPU trace 写入服务端 `profiler-config` 中指定的
`torch_profiler_dir`，不会写入 `--output-json` 所在目录。

### 推荐：按 batch 独立采集并归档

```bash
./examples/dsa_sparse/run_glm5_dsa_sparse_profiles.sh \
  --model-path /models/GLM-5.1-W4A8 \
  --output-root /data/dsa-profiles \
  --run-name mock-batch-scaling \
  --batch-sizes 1,2,4,8 \
  --prompt-tokens 10600 \
  --max-tokens 32 \
  --warmup-rounds 1 \
  --rounds 1 \
  --max-num-seqs 8 \
  --num-gpu-blocks-override 768
```

注意该 shell 脚本的 `--batch-sizes` 使用逗号分隔；Python 客户端使用空格分隔。

包装脚本会为每个 batch 独立启动和停止服务，从而缩短单次采集窗口，并将结果
归档为：

```text
<output-root>/<run-name>/<timestamp-pid>/
  bs1/
    trace/
    server.log
    result.json
    serve-command.txt
    profile-command.txt
  bs2/
    ...
```

包装脚本默认 `num_gpu_blocks_override=128`，只足够默认 batch 1。多 batch
采集必须根据 `max_batch_size * 81 + 1` 设置更大的值。额外的 vLLM 参数同样
放在 `--` 之后。

## 8. 离线解析 profiler 数据

默认只解析 rank 0，并导出适合 MindStudio Insight 打开的 DB：

```bash
python3 examples/dsa_sparse/parse_glm5_dsa_profile.py \
  /data/dsa-profiles/mock-batch-scaling/<run>/bs1/trace \
  --output-dir /data/dsa-profiles-parsed/bs1
```

指定多个 rank：

```bash
python3 examples/dsa_sparse/parse_glm5_dsa_profile.py \
  /data/dsa-profiles/mock-batch-scaling/<run>/bs1/trace \
  --output-dir /data/dsa-profiles-parsed/bs1 \
  --rank 0 \
  --rank 1
```

解析全部 rank 并导出 text：

```bash
python3 examples/dsa_sparse/parse_glm5_dsa_profile.py \
  /data/dsa-profiles/mock-batch-scaling/<run>/bs1/trace \
  --output-dir /data/dsa-profiles-parsed/bs1-text \
  --all-ranks \
  --export-type text \
  --max-processes 1
```

解析器会在输入目录向下最多三层查找包含 `FRAMEWORK` 或 `PROF_*` 的原始
profile 目录。`torch_npu.profiler.analyse` 会先在原始 profile 目录下生成
`ASCEND_PROFILER_OUTPUT`，脚本再把 DB 或 text 输出及 profiler metadata
复制到单独的 `--output-dir`。因此解析期间需要同时容纳原始数据、原地解析结果
和导出副本；脚本不会自动删除原始数据。

优先使用默认的 rank 0 DB。`--all-ranks --export-type text` 会显著增加解析时间
和磁盘占用。

## 常见问题定位

### 服务成功返回，但没有算子日志

依次检查：

1. prompt token 数是否严格大于 10368；
2. completion 是否至少生成 2 个 token；
3. 服务是否使用当前分支安装出的 `vllm_ascend`；
4. DSA general plugin、platform patch 和 runtime patch 是否加载；
5. 日志是否来自当前服务进程；
6. 是否误用了已经包含一次性算子标记的旧日志。

优先运行：

```bash
python3 examples/dsa_sparse/diagnose_glm5_dsa_sparse_env.py \
  --server-log /tmp/glm5-dsa.log
```

### `BinaryGetFunction failed` 或找不到 maintain kernel

先运行：

```bash
python3 check_asu_hbm_index_ops.py --diagnose-aicpu
```

重点检查 AICPU OPP 目录、`cust_aicpu_kernel.json`、
`libtransformer_aicpu_kernels.so`、`RunCpuKernel` 和注册名
`AsuHbmIndexMaintainAicpu`。

### profile 数据过大或解析时间过长

- 使用包装脚本按 batch 独立采集；
- 保持 `--warmup-rounds 1 --rounds 1`；
- 减少 `--max-tokens`，但不能小于 2；
- 默认只解析 rank 0；
- 优先使用 `--export-type db`；
- 不要在没有必要时使用 `--all-ranks`。

### benchmark 结果不可比较

不同测试间必须保持以下参数一致：

- `--miss-count`；
- warmup 和 timed iteration 数；
- batch size；
- NPU 型号、CANN、PyTorch NPU 和 custom-op 构建版本；
- 同一台机器上的其他 NPU 工作负载状态。

单算子 benchmark 测量的是直接 custom-op 调用的 NPU Event 时间，不包含 vLLM
调度、模型计算、状态恢复和 HTTP 开销，也不是 `npugraph_ex` 服务级性能。
