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
- 每次 warmup 和采样前恢复同一份索引状态；
- 状态恢复和恢复后的同步位于 NPU Event 计时区间之外；
- 每个 case 计时前都会先执行一次功能检查。

终端输出 `mean`、`median`、`p95`、`min` 和 `requests/s`。JSON 还包含
`max`、`query_items/s`、完整配置和每次迭代的原始毫秒采样。

当 `--miss-count 0` 时 maintain 会直接处理 `free_head == 0` 的场景，接近
空操作；测试实际淘汰开销时应使用非零值。

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

KV cache 分配完成后，框架会按 layer id 和 Indexer/nope/rope 的固定顺序，
把本地 tensor 地址与字节长度一次性传给 `aiv_init`。P 侧最后一个 prefill
chunk 会通过同步的 `aiv_put_batch + aiv_wait` 写入所有有效 prompt token，
包括 Indexer cache、MLA nope/rope cache 和最后一个非整块 tail。普通 DSA
Decode 中新完成的整块也沿用该路径。lookup miss 使用同步的
`aiv_get_batch + aiv_wait` 直接写入 resident physical slot。KVIO Python
接口只接收整数列表，因此 GET 热路径需要将一次批量地址元数据从 NPU 搬到
CPU。当前 KVIO 接口没有远端删除 API，请求结束只清理本地 pool entry 到
整数 request ID 的映射。

非 P/D 分离路径也会保留每层最后一个 Prefill query 的 TopK。第一次进入
sparse decode 时，每层先按 TopK score 顺序选取有效历史 token，排除当前
dense tail 和重复位置，再按历史位置升序补齐到 8192 个 resident token；
这些 token 依次写入 resident slot `[0, 8192)`。完成初始化后，本次 Decode
实时计算的 TopK 再进入正常的 lookup、miss load 和 maintain。因而 Prefill
TopK 只决定初始 resident membership，不替代 Decode 阶段的逐 token TopK。

### KVIO DSA 的 P/D 分离

P、D 两个实例都需要启用 DSA、选择 KVIO backend，并使用
`DSAKVIOConnector`。P 侧示例：

```json
{
  "kv_connector": "DSAKVIOConnector",
  "kv_role": "kv_producer",
  "engine_id": "dsa-prefill"
}
```

D 侧示例：

```json
{
  "kv_connector": "DSAKVIOConnector",
  "kv_role": "kv_consumer",
  "engine_id": "dsa-decode"
}
```

两侧的 `dsa_sparse_config` 都使用 `kv_backend="kvio"`，且
`kvio_model_id`、`block_size`、模型与 TP/PP layer 映射必须一致；
`kvio_pd_flag` 按 KVIO 部署对 P/D 角色的约定分别配置。D 侧必须关闭 prefix
caching，且两侧不能启用 speculative decoding 或 async scheduling。当前紧凑
初始化不支持把本地 prefix hit 与远端 DSA 状态混合：

```bash
--no-enable-prefix-caching \
--kv-transfer-config \
'{"kv_connector":"DSAKVIOConnector","kv_role":"kv_consumer","engine_id":"dsa-decode"}'
```

P 的最后一个 Prefill chunk 会复用每层 Lightning Indexer 已算出的最后一个
query token TopK；每个 worker 按全局 rank 回传本地 layer TopK，scheduler 在请求
结束前完成聚合。P 请求完成时，connector 在输出的 `kv_transfer_params` 中返回
protocol-v3 manifest、`dsa_kvio_layer_topk_by_rank` 和 `last_token_id`。
manifest 的 `state=READY` 只会在全部 P worker 已完成同步
`aiv_put_batch + aiv_wait`、且 scheduler 收齐对应 rank 的最后 Prefill TopK 后
发布。manifest 还携带 generation、P world size 和 cache layout fingerprint；
D 在分配 block 前校验协议版本、READY、模型/布局和 P/D 拓扑，不一致立即拒绝，
不会回退为部分读取。KVIO request id 使用 P 实例的 runtime `engine_id` 作为
命名空间；D 直接绑定 manifest 中的远端 id，因此 D 自己的 `engine_id` 不参与
远端 key 计算。

路由层仍负责控制面交接，不负责传输 KV tensor：

1. 发给 P 的请求必须设置
   `kv_transfer_params={"do_remote_decode": true, "do_remote_prefill": false}`
   并限制 `max_tokens=min_tokens=1`；
2. P 响应没有 `kv_transfer_params` 时按普通请求处理，不应强行发起紧凑 DSA
   handoff；这覆盖未完成 Prefill、短请求等情况；
3. P 响应有 `kv_transfer_params` 时，路由层必须原样传入 D，并把
   `last_token_id` 作为 P 已生成的首 token 追加到 D 的 token 序列。D 收到的
   token 数应等于 manifest 的 `stored_token_count + 1`；
4. 当前通用 proxy 示例只转发 `kv_transfer_params`，不会替业务请求追加 token
   id；接入层必须补上这一动作，并确保最终响应只向客户端返回一次 P 首 token。

manifest 包含稳定的 KVIO request id、远端有效 token 数、逻辑 block 数和
resident/tail 布局；逐层 TopK 是紧凑索引种子。D scheduler 再把自己分配的
dense Indexer block table 与 compact MLA resident block table 加入 worker
metadata。

D worker 在第一次模型 forward 前完成三步同步初始化：

1. 从 KVIO 加载全量 Indexer cache 到 D 的 dense Indexer block table；
2. 对每个本地 layer 先选入最后一个 Prefill token 的有效 TopK，再按历史 token
   位置升序、排除重复和 dense tail，确定性补齐到 8192 个 resident token；
3. 从 KVIO 加载这些逐层离散 MLA token 和非整块 tail 到 compact resident
   table，并初始化每层独立的 `token_to_slot`、`slot_to_token`、resident count
   和 prefill-ready 状态。resident slot 保留 TopK score 顺序，KVIO 直接按离散
   token 地址读取；补齐部分仍按历史 token 位置顺序排列。

追加到 D prompt 的 P 首 token 会在这次 forward 中直接按 sparse-decode query
处理，而不是退回 dense Prefill。之后 D Decode 的 Lightning Indexer 仍对完整
序列打分；历史 token 在 resident 索引中命中时直接使用本地 slot，miss 时通过
KVIO GET 写入 lookup 分配的 resident slot，再执行 maintain。当前协议只接受
严格大于 DSA sparse threshold
的 D 请求（默认 block size 128 时为 10368 token）。换言之，P 已存 prompt 加
首 token 后必须至少为 10369 token；更短请求不会发布 compact manifest。协议
假定 P/D 的对应并行 rank 连接到同一个可跨节点访问的 KVIO 数据面，且
`aiv_wait` 返回后 PUT 对 D 可见。

该路径不创建 Mooncake transport，也不需要 Mooncake 的 side-channel/握手元
数据；Indexer、resident MLA 初始化和后续 lookup miss 都通过 KVIO。外部路由
仍不可省略，因为它负责 P/D 选址、一次 Prefill、字段转发和首 token 衔接。
当前 `rdma_kv_ops` Python API 没有观察到远端 delete/TTL 调用，因此
`request_finished`/preempt/初始化失败只释放本地 request-to-pool 绑定和 resident
row。生产部署必须由 KVIO 服务配置 TTL/配额/后台回收；若 KVIO 不保证
`aiv_wait` 后跨节点可见性，也不能把 `state=READY` 当成可读承诺。

### Mooncake 初始传输 + KVIO 持久化

如果现有 P/D 部署已经使用 Mooncake，可将两侧 connector 改为
`DSAMooncakeConnector`，无需修改
`kv_p2p/mooncake_connector.py` 或 Mooncake 库。P 侧示例：

```json
{
  "kv_connector": "DSAMooncakeConnector",
  "kv_role": "kv_producer",
  "kv_port": 30000,
  "engine_id": "dsa-prefill",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {"dp_size": 1, "tp_size": 16},
    "decode": {"dp_size": 1, "tp_size": 16}
  }
}
```

D 侧使用不同的本地 `kv_port` 和 `engine_id`，其余拓扑描述保持一致：

```json
{
  "kv_connector": "DSAMooncakeConnector",
  "kv_role": "kv_consumer",
  "kv_port": 30100,
  "engine_id": "dsa-decode",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {"dp_size": 1, "tp_size": 16},
    "decode": {"dp_size": 1, "tp_size": 16}
  }
}
```

该模式仍要求两侧 `dsa_sparse_config.kv_backend="kvio"`：KVIO 保存完整
prompt KV，并负责 D 后续 lookup miss；Mooncake 只负责一次初始 handoff。
当前适配器要求 P/D 的 TP、DP 和 world size 一致，且 PP=PCP=DCP=1。
D 侧仍需关闭 prefix caching、async scheduling 和 speculative decoding。

P 结束后会延迟释放原始 dense block。D 在第一次 forward 前直接通过
Mooncake TransferEngine 从这些 block 拉取：

1. 全量 Indexer cache；
2. 每层确定性的 8192 个 resident MLA token；
3. 最后一个非整块 dense tail。

中间 2048 个 lookup 空闲槽不走网络。适配器按真实的 Indexer/MLA tensor
地址和各自 block table 生成 scatter/gather 描述，不进入现有
`MooncakeConnector` 不支持 DSA 异构 block 数的 HMA 分支。D 拉取成功并收到
Mooncake ACK 后，P 才释放延迟 block；D 随后只绑定 manifest 中的 KVIO request
id、初始化 lookup 映射，不再从 KVIO 重复加载初始 resident 数据。

路由层沿用上面的同一控制面约定：原样转发整个 `kv_transfer_params`，并把
`last_token_id` 追加到 D 的 token 序列。新增字段都位于标准
`kv_transfer_params` 内；若现有路由已经透明转发 Mooncake 参数并完成首 token
衔接，不需要增加新的数据传输逻辑。

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
