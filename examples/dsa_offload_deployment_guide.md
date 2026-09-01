# GLM-5.2 DSA Offload 部署指南

本文面向 `dsa-offload-0.23` 分支，说明如何使用当前框架侧实现启动
GLM-5.2 服务，以及 DSA Offload 新增参数的含义、约束和推荐配置。

本文核对基线为当前 checkout 的 `dsa-offload-0.23`。如果分支继续演进，应重新
核对本文列出的 gate 和参数契约。

> 本文命令以单个 Prefill engine 和单个 Decode engine 为例。模型需要多少
> NPU、TP 应设为多少，仍取决于权重格式和单卡显存；调整 TP 时，必须同时
> 修改命令行和 `kv_connector_extra_config` 中的 TP。

## 1. 当前实现的边界

启用 DSA Offload 后，框架侧的数据路径是：

```text
Prefill Main KV Cache
  |-- 完整 block --------------------> IO backend (mock 或 KVIO)
  `-- 未对齐的末尾 token
       |-- 分离 P/D -----------------> Mooncake 或 LocalShm connector
       `-- 单 engine kv_both --------> 本地 Main -> Hot Cache 复制

Decode
  |-- 分离 P/D：connector 接收 handoff 和末尾 token
  |-- 单 engine：Prefill 完成后本地晋升为 Hot Cache 请求
  |-- DSA TopK 查找
  |-- cache miss 从 IO backend 读入 Hot Cache
  `-- SFA 从 Hot Cache/Main Cache 读取
```

需要特别区分两个 backend：

- `mock`：默认用于框架链路冒烟。它会跳过完整 block 的 PUT/GET，能检查服务
  启动、P/D 控制面、DSA lookup 和调度路径，但不能验证真实 offload 数据面，
  也不能据此判断长上下文精度。
- `kvio`：真实外部容量层数据面。它会调用 `rdma_kv_ops.aiv_init`、
  `npu_get_put_batch` 和 `npu_send_wait`，正式部署和精度验证必须使用它。

当前分支有以下硬约束：

| 项目 | 当前要求 |
| --- | --- |
| 硬件 | Ascend A5/950 系列 |
| 模型 | `model_type=glm_moe_dsa`，即 GLM-5 系列 |
| DSA TopK | 模型配置中 `index_topk=2048` |
| 最大上下文 | `--max-model-len` 不得超过 `131072` |
| KV connector | 分离 P/D 使用 `MooncakeConnectorV1` 或 `LocalShmConnector`；单 engine 可不配置 |
| P/D TP | Prefill TP 和 Decode TP 必须相等 |
| PP/PCP/DCP | 均必须为 1，即当前不能启用 |
| speculative decoding | 只支持 `method=mtp`，draft token 数最多 15 |
| Decode preemption | 当前不支持；容量规划应避免触发 preemption |

因此，普通 GLM-5.2 文档中的 `PP2`、DCP、P/D 不同 TP、
`MultiConnector`、超过 128K 的上下文，以及 `method=deepseek_mtp`，都不能
直接复制到本分支的 DSA Offload 命令中。

这里的“不配置 connector”只表示同一个 engine 同时执行 Prefill 和 Decode。
两个独立 engine 之间必须有 connector 携带请求 handoff 和未满 block 的尾部
payload；不能仅靠 IO backend 代替这段 P/D 协议。

## 2. 安装和启动前检查

### 2.1 安装当前 checkout

在每个 P/D 节点使用同一分支、同一提交和同一依赖环境：

```bash
cd /home/solidyang/workspace/vllm-ascend
git branch --show-current
git rev-parse HEAD
git submodule update --init --recursive

COMPILE_CUSTOM_KERNELS=1 pip install --no-build-isolation -e .
```

安装后确认 Python 实际加载的是当前 checkout，并检查 DSA 自定义算子：

```bash
python3 - <<'PY'
from pathlib import Path
import importlib
import torch
import vllm_ascend

repo = Path("/home/solidyang/workspace/vllm-ascend").resolve()
package = Path(vllm_ascend.__file__).resolve()
assert repo in package.parents, (repo, package)

importlib.import_module("vllm_ascend.vllm_ascend_C")
required = [
    "dsa_offload_lookup_update_batch",
]
missing = [name for name in required if not hasattr(torch.ops._C_ascend, name)]
assert not missing, f"missing _C_ascend ops: {missing}"
print(f"vllm_ascend={package}")
print("DSA custom ops: OK")
PY
```

使用 `kvio` 前还要检查 KVIO Python 模块和 native op：

```bash
python3 - <<'PY'
import importlib
import torch
import vllm_ascend.vllm_ascend_C  # noqa: F401

rdma_kv_ops = importlib.import_module("rdma_kv_ops")
assert hasattr(rdma_kv_ops, "aiv_init")
for name in ("npu_get_put_batch", "npu_send_wait"):
    assert hasattr(torch.ops._C_ascend, name), name
print("KVIO Python module and native ops: OK")
PY
```

### 2.2 检查模型配置

以下检查同时兼容顶层配置和 `text_config`：

```bash
MODEL=/path/to/GLM-5.2-w8a8c8

python3 - "$MODEL" <<'PY'
import json
import pathlib
import sys

config = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
text = config.get("text_config") or config
assert text.get("model_type") == "glm_moe_dsa", text.get("model_type")
assert text.get("index_topk") == 2048, text.get("index_topk")
print({
    "model_type": text.get("model_type"),
    "index_topk": text.get("index_topk"),
    "hidden_size": text.get("hidden_size"),
    "num_nextn_predict_layers": text.get("num_nextn_predict_layers", 0),
})
PY
```

如果使用裁剪过的 tiny 模型，需保证它仍满足 A5 算子的 shape 约束。已知某些
构建会拒绝 `hidden_size=8` 的 `MlaPrologV3`；此时应换成硬件可执行的小模型，
而不是把失败归因于 DSA Offload 控制面。

### 2.3 网络环境

每个节点按自己的服务网卡设置，`NODE_IP` 必须是其他节点可达的地址：

```bash
NODE_IP=192.168.1.10
NIC_NAME=eth0

export VLLM_HOST_IP="$NODE_IP"
export HCCL_IF_IP="$NODE_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export MC_TCP_BIND_ADDRESS="$NODE_IP"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
```

Prefill、Decode 和代理之间的 HTTP 端口，以及 Mooncake 的 `kv_port` 均需互通。
多 rank/多实例部署时，应为各实例预留互不冲突的端口范围。

## 3. 新增参数怎么填

### 3.1 `--additional-config`

最小配置如下：

```json
{
  "dsa_offload": {
    "io_backend": "mock",
    "kvio_model_id": 52
  },
  "ascend_compilation_config": {
    "enable_npugraph_ex": false
  }
}
```

| 字段 | 是否必填 | 推荐值和含义 |
| --- | --- | --- |
| `dsa_offload` | 是 | 存在该对象即启用 DSA Offload |
| `dsa_offload.io_backend` | 否 | 默认 `mock`；首次冒烟用 `mock`，真实部署填 `kvio` |
| `dsa_offload.kvio_model_id` | 否 | 非负整数，默认 0；同一模型的 P/D 必须一致，不同并行服务建议使用不同 ID |
| `ascend_compilation_config.enable_npugraph_ex` | 否 | 不是 DSA 新参数；首次验证建议为 `false`，稳定后再单独验证图模式 |

`enable_sparse_sfa_c8`、`enable_sparse_li_c8` 和
`multistream_overlap_shared_expert` 是已有 GLM/Ascend 优化项，不是启用 DSA
Offload 的必要条件。建议先用最小配置打通，再逐项打开并回归精度、显存和性能。

GLM-5.2 的 OpenAI API 能力也不是 DSA 新参数。如需自动工具调用和 reasoning
解析，可在 P/D 两侧保持一致地增加：

```bash
--enable-auto-tool-choice \
--tool-call-parser glm47 \
--reasoning-parser glm45
```

本文命令为了缩小首轮验证变量，暂未启用这些 API 层选项。

### 3.2 `--kv-transfer-config` 与 connector 选择

有三种部署方式：

| 拓扑 | 配置方式 | 适用范围 |
| --- | --- | --- |
| 跨主机或通用分离 P/D | `MooncakeConnectorV1` | 生产 P/D 数据传输 |
| 同一主机、同一容器内的分离 P/D | `LocalShmConnector` | 简化单机验证，不支持跨主机 |
| 单 engine 混合 Prefill/Decode | 完全省略 `--kv-transfer-config` | 本地 `kv_both`，无需 P/D connector |

分离 P/D 的 Prefill 侧 Mooncake 示例：

```json
{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": 30000,
  "engine_id": "glm52-dsa-prefill",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "prefill": {"dp_size": 1, "tp_size": 8},
    "decode": {"dp_size": 1, "tp_size": 8}
  }
}
```

Decode 侧只需把 `kv_role` 改为 `kv_consumer`，并使用独立的 `kv_port` 和
`engine_id`。

| 字段 | 填法 |
| --- | --- |
| `kv_connector` | 分离 P/D 填 `MooncakeConnectorV1` 或 `LocalShmConnector`，不能填 `MultiConnector` |
| `kv_role` | 分离 P/D 的 P=`kv_producer`、D=`kv_consumer`；无 connector 时由框架隐式使用 `kv_both` |
| `kv_port` | connector 控制配置的基础端口；实例之间不能冲突 |
| `engine_id` | 每个 engine 使用唯一字符串 |
| `kv_load_failure_policy` | bring-up 推荐 `fail`，便于暴露真实传输失败 |
| `prefill.dp_size` / `decode.dp_size` | 示例是单 P、单 D engine，所以均为 1；按真实全局拓扑填写 |
| `prefill.tp_size` / `decode.tp_size` | 两者必须相等，且必须和两侧 `--tensor-parallel-size` 一致 |

`--block-size` 也必须在 P/D 两侧保持一致。当前 handoff 会在运行时校验它，
本文统一使用 `128`。

同机 LocalShm 的配置只需替换 connector，并增加三个可选字段：

```json
{
  "kv_connector": "LocalShmConnector",
  "kv_role": "kv_producer",
  "kv_port": 30000,
  "engine_id": "glm52-dsa-prefill",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "prefill": {"dp_size": 1, "tp_size": 8},
    "decode": {"dp_size": 1, "tp_size": 8},
    "shm_dir": "/dev/shm/vllm-ascend-local-kv",
    "shm_namespace": "glm52-dsa-instance-a",
    "shm_timeout": 120
  }
}
```

Decode 侧使用完全相同的 `shm_dir` 和 `shm_namespace`，只把 role、port 和
engine ID 改成 Decode 值。LocalShm 还有以下限制：

- P/D 必须在同一主机和同一容器/IPC namespace 中，且能看到同一个绝对
  `shm_dir`。
- P/D TP 必须相等，rank `r` 只与 rank `r` 交换数据；DP、PP、PCP、DCP
  必须为 1。
- payload 通过文件映射同步执行 D2H/H2D。它不创建 Mooncake
  TransferEngine，也没有 TP 重分片或后台传输线程。
- 每个独立部署使用唯一 `shm_namespace`，避免请求文件命名空间冲突。
- `shm_timeout` 单位为秒且必须大于 0；默认值为 120。

单 engine 模式不要构造 `kv_role=kv_both` 的 LocalShm 配置，而是完全省略
`--kv-transfer-config`。框架会在最终 Prefill step 后，将请求从 Main cache
本地晋升到 Hot Cache，并直接复制未满 block 的尾部。

## 4. 推荐验证顺序

### 4.1 先运行现有 probe

用小模型、两张卡验证 P/D 基本路径，默认 backend 就是 `mock`：

```bash
cd /home/solidyang/workspace/vllm-ascend

bash examples/dsa_offload_probe.sh \
  --model /path/to/hardware-compatible-glm-moe-dsa \
  --host-ip 192.168.1.10 \
  --ifname eth0 \
  --connector mooncake \
  --io-backend mock \
  --scenario pd \
  --verify-path
```

然后切到真实 KVIO：

```bash
bash examples/dsa_offload_probe.sh \
  --model /path/to/hardware-compatible-glm-moe-dsa \
  --host-ip 192.168.1.10 \
  --ifname eth0 \
  --connector mooncake \
  --io-backend kvio \
  --kvio-model-id 52 \
  --scenario pd \
  --verify-path
```

probe 是单机 TP1 的功能验证工具，不替代 GLM-5.2 多卡容量验证。

同一容器内验证 LocalShm 时改为 `--connector local-shm`；验证无 connector
的本地混合生命周期时使用 `--scenario both --connector none`。

### 4.2 GLM-5.2 P/D 部署变量

下面以 P、D 各一个 TP8 engine 为例。若使用非量化权重，删除
`--quantization ascend`；w8a8c8/w4a8c8 权重保留该参数。

两侧使用相同的基础变量：

```bash
MODEL=/models/GLM-5.2-w8a8c8
SERVED_MODEL=glm-5.2
TP_SIZE=8
BLOCK_SIZE=128
MAX_MODEL_LEN=131072
KVIO_MODEL_ID=52

# 第一次打通框架路径用 mock；真实 offload 改成 kvio。
IO_BACKEND=mock

DSA_CONFIG=$(printf \
  '{"dsa_offload":{"io_backend":"%s","kvio_model_id":%d},"ascend_compilation_config":{"enable_npugraph_ex":false}}' \
  "$IO_BACKEND" "$KVIO_MODEL_ID")
```

### 4.3 启动 Prefill

在 Prefill 节点执行。`ASCEND_RT_VISIBLE_DEVICES` 的设备数应与 `TP_SIZE`
一致：

```bash
PREFILL_KV_PORT=30000
PREFILL_KV_CONFIG=$(printf \
  '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":%d,"engine_id":"glm52-dsa-prefill","kv_load_failure_policy":"fail","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":%d},"decode":{"dp_size":1,"tp_size":%d}}}' \
  "$PREFILL_KV_PORT" "$TP_SIZE" "$TP_SIZE")

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8100 \
  --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --quantization ascend \
  --trust-remote-code \
  --block-size "$BLOCK_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"NONE"}' \
  --additional-config "$DSA_CONFIG" \
  --kv-transfer-config "$PREFILL_KV_CONFIG"
```

### 4.4 启动 Decode

在 Decode 节点设置它自己的 `NODE_IP`/网卡环境后执行：

```bash
DECODE_KV_PORT=30100
DECODE_KV_CONFIG=$(printf \
  '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":%d,"engine_id":"glm52-dsa-decode","kv_load_failure_policy":"fail","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":%d},"decode":{"dp_size":1,"tp_size":%d}}}' \
  "$DECODE_KV_PORT" "$TP_SIZE" "$TP_SIZE")

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8200 \
  --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --quantization ascend \
  --trust-remote-code \
  --block-size "$BLOCK_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"NONE"}' \
  --additional-config "$DSA_CONFIG" \
  --kv-transfer-config "$DECODE_KV_CONFIG"
```

`max_num_seqs` 会直接影响 Decode Hot Cache 的固定显存。初次部署从小值开始，
确认显存余量后再增加；如果出现 `DSA Offload fixed cache exceeds available KV
memory`，优先降低 `--max-num-seqs`，再检查模型权重、TP 和
`--gpu-memory-utilization` 的容量规划。

`--no-enable-prefix-caching` 只关闭 vLLM 的跨请求前缀复用。DSA Offload 会独立
生成外存 storage key 所需的 block hash，因此关闭 prefix cache 不会关闭
offload，也不要求通过 connector 间接启用 block hasher。

### 4.5 启动 P/D 代理并请求

待 P/D `/health` 均返回成功后启动标准 P/D 代理：

```bash
cd /home/solidyang/workspace/vllm-ascend

python3 examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prefiller-hosts 192.168.1.10 \
  --prefiller-ports 8100 \
  --decoder-hosts 192.168.1.11 \
  --decoder-ports 8200
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "请简要解释什么是稀疏注意力。"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

`mock` 阶段只要求服务、请求和预期框架路径正常；切换 `kvio` 后才进行重复
prompt、block 对齐/非对齐长度、并发请求和长上下文精度验证。

## 5. 单服务 `kv_both` 部署

不需要 P/D 代理时，可由一个 engine 同时完成 Prefill 和 Decode。当前实现会
隐式使用 `kv_both`，命令中不要传 `--kv-transfer-config`：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --quantization ascend \
  --trust-remote-code \
  --block-size "$BLOCK_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"NONE"}' \
  --additional-config "$DSA_CONFIG"
```

本地晋升仍依赖 IO backend 恢复完整历史 block：`mock` 只验证调度、lookup、
Main-to-Hot 尾部复制和 SFA 控制路径；需要验证输出精度时必须改用 `kvio`。

## 6. 可选：启用 MTP

不使用 MTP 时完全省略 `--speculative-config`。启用时当前分支必须写成：

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

注意：

- 不能沿用普通 GLM-5.2 示例中的 `method=deepseek_mtp`。
- `num_speculative_tokens` 范围是 1 到 15。
- 模型配置必须包含 next-token prediction layer。
- P/D 模式下，Prefill 可配置 1 个 draft token，Decode 配置实际目标值；先在
  无 MTP 模式打通，再单独打开。

## 7. 从 `mock` 切换到 `kvio`

正式切换只需把两侧相同的配置改为：

```bash
IO_BACKEND=kvio
KVIO_MODEL_ID=52

DSA_CONFIG=$(printf \
  '{"dsa_offload":{"io_backend":"%s","kvio_model_id":%d},"ascend_compilation_config":{"enable_npugraph_ex":false}}' \
  "$IO_BACKEND" "$KVIO_MODEL_ID")
```

但切换前必须同时满足：

1. P/D 两侧已安装 `rdma_kv_ops`，且 `aiv_init` 可见。
2. 当前 `_C_ascend` 同时注册了 `npu_get_put_batch` 和 `npu_send_wait`。
3. P/D 使用相同的 `kvio_model_id`、模型权重、TP、block size 和 cache layout。
4. KVIO 依赖的 RDMA/容量层已按其部署要求初始化且节点互通。
5. 日志中没有 `aiv_init failed`、PUT/GET 或 `npu_send_wait` 错误。

同一套 KVIO 服务中，不同模型或相互隔离的部署建议使用不同的
`kvio_model_id`，避免共享同一模型命名空间。

## 8. 常见失败和处理

| 错误或现象 | 检查项 |
| --- | --- |
| `DSA Offload requires Ascend A5` | 当前节点不是 A5/950，或设备识别环境不正确 |
| `supports only the GLM-5 family` | 模型的 `model_type` 不是 `glm_moe_dsa` |
| `requires index_topk=2048` | 权重对应的 `config.json` 不符合当前算子契约 |
| `max_model_len must not exceed 131072` | 降低 `--max-model-len`，不能直接使用 GLM-5.2 的 198K/256K/1M 示例 |
| `supports only MooncakeConnectorV1 or LocalShmConnector` | 分离 P/D 不要使用 `MultiConnector` 或其他 connector |
| `omit kv_transfer_config for local kv_both` | 单 engine 完全删除 `--kv-transfer-config`，不要给 LocalShm 配 `kv_both` |
| `requires equal Prefill and Decode TP sizes` | 对齐命令行 TP 和 connector 内的两处 TP |
| `does not support PP, PCP, or DCP` | 删除 PP、PCP、DCP 参数，均保持 1 |
| `supports speculative decoding only with MTP` | 使用 `method=mtp`，或先删除 speculative 配置 |
| `fixed cache exceeds available KV memory` | 降低 `max_num_seqs`，检查 TP 和显存规划 |
| 缺少 `dsa_offload_lookup_update_batch` | 当前加载的是旧 `.so`；重新编译安装并检查 Python 包路径 |
| `No module named rdma_kv_ops` | KVIO 运行时未安装；mock 不需要该模块 |
| `KVIO aiv_init failed` | 检查 KVIO/RDMA 初始化、注册内存、设备和 native 库版本 |
| Mooncake 传输失败或超时 | 检查 `VLLM_HOST_IP`、NIC、`MC_TCP_BIND_ADDRESS`、端口和防火墙 |
| LocalShm 等待 manifest 超时 | 检查 P/D 是否共享 IPC namespace、目录/namespace 是否一致、Prefill 是否仍存活 |
| tiny 模型在 `MlaPrologV3` 失败 | 检查 shape，换用 A5 算子支持的小模型 |
| Decode 出现 preemption | 当前实现不支持；降低并发/序列长度或增加容量 |

## 9. 上线验收清单

建议按下面顺序记录结果，避免把控制面成功误当成数据面成功：

1. 环境：两侧分支、HEAD、vLLM/vLLM Ascend/CANN 版本一致。
2. 启动：P/D 或 `kv_both` 均通过 `/health`。
3. 生命周期：分离 P/D handoff 或单 engine 本地晋升完成，无等待或超时。
4. DSA 算子：lookup/update 单请求和 batch 路径均实际执行。
5. 尾部 payload：Mooncake、LocalShm 或本地 Main-to-Hot 的未对齐末尾 token
   能正确传输。
6. KVIO payload：完整 block PUT、Decode miss GET 和 wait 均实际执行。
7. 功能：覆盖 block 对齐、非对齐、重复历史、短/长上下文和并发请求。
8. 精度：使用 `kvio` 与不启用 DSA Offload 的基线比较输出；不要使用
   `mock` 做精度结论。
9. 稳定性：持续负载下无 preemption、内存增长、KVIO/Mooncake 错误。
10. 性能：功能稳定后，再逐项启用 C8、共享专家多流和图模式并分别回归。
