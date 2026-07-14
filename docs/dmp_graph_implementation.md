# DMP Full Decode ACL Graph 实现说明

## 1. 文档范围

本文记录当前 DMP（Dual Microbatch Pipeline）接入 vLLM-Ascend
`FULL_DECODE_ONLY` ACL Graph 的实现方式，以及相关 Python 代码的修改位置。

当前代码基于 `releases/v0.18.0`，对应的 DMP 功能提交为：

```text
826cdb9c8 feat(dmp): add dual-microbatch decode graph path
```

本次没有修改 `csrc`、自定义算子源码或 CANN 算子实现。

## 2. 原始问题

原来的 ACL Graph 开关可以正常开启，但是 Graph capture 时没有
`DMPContext`，因此模型执行的是原始 forward：

```text
Graph capture
  -> dmp_context is None
  -> original model forward
  -> 捕获普通单流 Graph
```

正式 decode 时即使临时创建了 `DMPContext` 也已经太晚。ACL Graph replay
直接重放捕获的 NPU 操作，不会再次执行 Python 模型 forward，因此无法在
replay 时从普通 forward 切换到 `dmp_forward()`。

## 3. 实现思路

实现没有重写 ACL Graph，而是复用 vLLM-Ascend 已有的
`ACLGraphWrapper` 和 `CUDAGraphMode.FULL_DECODE_ONLY`。主要工作是让 DMP
上下文和双流调度进入现有的 capture/replay 生命周期。

修改后的流程如下：

```text
Full-decode Graph warmup
  -> 判断 batch 是否满足 DMP 静态条件
  -> 按 BatchDescriptor 创建 DMPContext
  -> 切分 microbatch A/B 的 attention metadata
  -> 创建并缓存第二条 NPU Stream
  -> 在 capture 前创建全部跨流 Event
  -> 将 DMPContext 写入 forward_context
  -> 执行 dmp_forward(graph_capture=True)
  -> ACLGraph 捕获 S0/S1 上的 NPU 操作和 Event 依赖
  -> 按 BatchDescriptor 缓存 DMPContext

正式 decode
  -> 根据 BatchDescriptor 取得已缓存的 DMPContext
  -> ACLGraph replay
```

不同 capture size（当前为 2、4、8、16）分别持有自己的
`DMPContext`、切片 metadata 和 Event，保证 capture 与 replay 使用同一批
静态对象。

## 4. 当前双流调度

当前 Graph 中的双流调度为：

```text
S1: microbatch A Indexer
S0: microbatch B Indexer
S0: 等待 A Indexer 完成
S0: microbatch A Sparse Attention
S0: microbatch B Sparse Attention
S0: microbatch A/B 合并 MLP
```

当前实现属于“同层双流”：A/B Indexer 可以位于两条 Stream，但
`forward_mlp_two_mb_once()` 仍然合并执行 A/B MLP。完整的跨层流水，也就是
`MB-A Indexer(L+1)` 与 `MB-B MLP(L)` 重叠，当前尚未启用。

## 5. Graph 成功的关键修改

### 5.1 Capture 阶段创建 DMPContext

文件：`vllm_ascend/worker/model_runner_v1.py`

位置：`NPUModelRunner._dummy_run()`

```python
def _dummy_run(..., is_graph_capturing=False, ...):
    # Full-decode Graph capture 时提前创建 DMPContext。
    if ...:
        dmp_context = self._get_or_create_dmp_graph_context(...)

    # 将 DMPContext 带入实际被 capture 的模型 forward。
    with set_ascend_forward_context(
        ...,
        dmp_context=dmp_context,
    ):
        ...
```

同一函数还在 full-graph warmup 阶段创建普通 DMPContext，使 DMP 相关 kernel
和 buffer 在正式 capture 前完成预热。

### 5.2 按 Graph shape 持久化上下文

文件：`vllm_ascend/worker/model_runner_v1.py`

位置：`NPUModelRunner.__init__()`、
`NPUModelRunner._get_or_create_dmp_graph_context()`

```python
class NPUModelRunner(...):
    def __init__(self, ...):
        # 第二条 Stream 只创建一次并重复使用。
        self._dmp_load_stream = ...

        # 每种 BatchDescriptor 对应一个持久化 Graph 上下文。
        self._dmp_graph_contexts = {...}

    def _get_or_create_dmp_graph_context(self, batch_descriptor, ...):
        # 查找或创建 DMPContext。
        ...

        # 在进入 capture 前准备 Event。
        dmp_context.prepare_graph_events(...)

        # 保持 metadata、Stream 和 Event 的生命周期。
        self._dmp_graph_contexts[batch_descriptor] = dmp_context
        ...
```

### 5.3 Replay 阶段复用 capture 上下文

文件：`vllm_ascend/worker/model_runner_v1.py`

位置：`NPUModelRunner.execute_model()`

```python
def execute_model(...):
    if ... and cudagraph_mode == CUDAGraphMode.FULL:
        # 不能在 replay 前重新创建上下文，必须使用 capture 时的对象。
        dmp_context = self._dmp_graph_contexts.get(batch_desc)
    else:
        ...

    with set_ascend_forward_context(
        ...,
        dmp_context=dmp_context,
    ):
        ...
```

### 5.4 将 DMPContext 写入 forward_context

文件：`vllm_ascend/ascend_forward_context.py`

位置：`set_ascend_forward_context()`

```python
def set_ascend_forward_context(
    ...,
    dmp_context=None,
):
    ...

    # dmp_forward 和 MoE/MLA 路径从这里取得当前 microbatch 上下文。
    forward_context.dmp_context = dmp_context
    ...
```

### 5.5 预创建并缓存跨流 Event

文件：`vllm_ascend/worker/dmp_context.py`

位置：`DMPContext.get_event()`、`DMPContext.prepare_graph_events()`

```python
@dataclass
class DMPContext:
    # 保证 Event 在 capture/replay 期间一直存活。
    _event_cache = {...}

    def get_event(self, tag):
        # 相同 tag 始终返回同一个 Event。
        ...

    def prepare_graph_events(self, num_layers):
        # 必须在进入 Graph capture scope 前创建。
        self.get_event("dmp_fork")
        for layer_idx in range(num_layers):
            self.get_event(...)
            ...
```

如果在 capture 中临时创建 Event，或者 capture 后销毁 Event，都会增加
多流 Graph capture/replay 不稳定的风险。

### 5.6 建立 S0/S1 的 capture 依赖

文件：`vllm_ascend/patch/worker/patch_deepseek_mtp.py`

位置：`dmp_forward()`

```python
def dmp_forward(...):
    dmp_ctx = get_forward_context().dmp_context
    ...

    # S0 是 ACL Graph 的主 capture Stream。
    s0 = torch.npu.current_stream()

    # S1 是 DMP 的第二条 Stream。
    s1 = dmp_ctx.kv_loader.load_stream

    # 将 S1 fork 到当前 Graph capture 拓扑中。
    fork_event = dmp_ctx.get_event("dmp_fork")
    fork_event.record(s0)
    s1.wait_event(fork_event)

    for layer_idx, layer in ...:
        # microbatch A Indexer 在 S1。
        with torch.npu.stream(s1), dmp_ctx.enter_microbatch(0):
            ...
            indexer_a_done.record(s1)

        # microbatch B Indexer 在 S0。
        with torch.npu.stream(s0), dmp_ctx.enter_microbatch(1):
            ...

        # S0 消费 A 的结果前等待 S1。
        s0.wait_event(indexer_a_done)
        ...

        # 下一层 S1 必须等待本层 MLP 输出。
        previous_layer_done.record(s0)
        ...
```

这里的 `fork_event` 是多流 Graph capture 的关键。它让第二条 Stream 成为主
capture Stream 的依赖分支，而不是一个与当前 Graph 无关的独立 Stream。

### 5.7 避开 TorchDynamo 对双流 Python 控制流的追踪

文件：`vllm_ascend/patch/worker/patch_deepseek_mtp.py`

位置：DMP patch 初始化、`dmp_forward()`

```python
if envs.VLLM_ASCEND_ENABLE_DMP:
    # Stream、Event 和 Python 分支不交给 TorchDynamo fullgraph 追踪。
    setattr(DeepseekV2Model, IGNORE_COMPILE_KEY, True)

    @torch.compiler.disable
    def dmp_forward(...):
        ...
```

这里只跳过模型级 TorchDynamo 编译。外层 `ACLGraphWrapper` 仍然负责捕获
实际 NPU 操作，因此不会关闭 ACL Graph。

### 5.8 Capture 时跳过动态 KV 加载

文件：`vllm_ascend/patch/worker/patch_deepseek_mtp.py`

位置：`dmp_forward()`

```python
def dmp_forward(...):
    is_capturing = get_forward_context().capturing
    ...

    if not is_capturing:
        # 包含动态长度、Host IO 和异步加载。
        ... classify_topk_indices(...)
        ... async_load_blocks(...)
        ... wait_load_complete(...)

    # capture 路径只记录静态可捕获的计算。
    ...
```

这是当前 Graph 能够稳定 capture 的必要限制。动态 KV classify/load 尚未被
纳入 Graph。

### 5.9 限制 DMP Graph 的静态输入条件

文件：`vllm_ascend/worker/model_runner_v1.py`

位置：`NPUModelRunner._is_dmp_eligible()`、
`NPUModelRunner._maybe_create_dmp_slices()`

```python
def _is_dmp_eligible(self, attn_metadata, num_input_tokens):
    # 当前仅支持 DecodeOnly。
    ...

    # token 数必须能稳定切分为两个 microbatch。
    ...

    # 当前仅支持每个 request 一个 decode token。
    ...

    # DSA-CP 暂不进入 DMP Graph。
    ...

def _maybe_create_dmp_slices(self, ...):
    # 创建 A/B token 范围。
    ...

    # 切分 attention metadata，并在需要时补齐 shape。
    ...
```

不满足条件的 Prefill、ChunkedPrefill、奇数 batch 和 DSA-CP 会回退到原始
forward，不会强行进入 DMP Graph。

## 6. DMP 计算路径的其他代码修改

以下修改是 DMP 模型路径和 microbatch 执行所需的支持代码。它们不是
ACL Graph 框架本身，但 Graph capture 会调用这些路径。

### 6.1 Microbatch 状态切换与输出合并

文件：`vllm_ascend/worker/dmp_context.py`

位置：`DMPSlice`、`DMPContext.enter_microbatch()`、
`DMPContext.slice_hidden_states()`、`DMPContext.merge_hidden_states()`

```python
class DMPSlice:
    # 保存 microbatch 的 token 范围和 padding 信息。
    ...

class DMPContext:
    def enter_microbatch(self, idx):
        # 临时替换 attention metadata、token 数和 MC2 mask。
        ...

    def slice_hidden_states(self, hidden_states, idx):
        ...

    def merge_hidden_states(self, hs_a, hs_b, output):
        # 去掉 dummy padding 后恢复原始 token 顺序。
        ...
```

### 6.2 将 SFA 拆成 Indexer 和 Sparse Attention

文件：`vllm_ascend/attention/sfa_v1.py`

位置：`AscendSFAImpl.forward_indexer_only()`、
`AscendSFAImpl.forward_sparse_attn_only()`

```python
class AscendSFAImpl(...):
    def forward_indexer_only(self, ...):
        # 只执行 Indexer 前处理、KV 写入和 top-k 选择。
        ...

    def forward_sparse_attn_only(self, indexer_result, ...):
        # 使用 Indexer 输出执行 Sparse Attention 和投影。
        ...
```

如果不拆分，完整 Attention 是一次整体调用，A/B Indexer 无法放入两条
Stream。

### 6.3 MLA Python custom-op 包装

文件：`vllm_ascend/ops/mla.py`

位置：`AscendMLA.forward_indexer_only()`、
`AscendMLA.forward_sparse_attn_only()`、`mla_forward_indexer_only()`、
`mla_forward_sparse_attn_only()` 及其 fake implementation 和注册代码。

```python
class AscendMLA(...):
    def forward_indexer_only(self, ...):
        return torch.ops.vllm.mla_forward_indexer_only(...)

    def forward_sparse_attn_only(self, ...):
        torch.ops.vllm.mla_forward_sparse_attn_only(...)
        ...

def mla_forward_indexer_only(...):
    # 从 forward_context 取得当前 microbatch metadata。
    ...

def mla_forward_sparse_attn_only(...):
    ...

# 注册 Python custom op；没有修改底层 C++/CANN 算子源码。
direct_register_custom_op(...)
```

### 6.4 Decoder Layer 和模型 forward patch

文件：`vllm_ascend/patch/worker/patch_deepseek_mtp.py`

位置：以下辅助函数及 patch 绑定：

```python
def forward_indexer_only(...):
    ...

def forward_sparse_attn_only(...):
    ...

def forward_mlp_only(...):
    ...

def forward_mlp_two_mb_once(...):
    # 当前仍然合并 A/B MLP。
    ...

def dmp_forward(...):
    ...

# 将辅助阶段和 dmp_forward 绑定到上游模型类。
DeepseekV2DecoderLayer.forward_indexer_only = ...
DeepseekV2DecoderLayer.forward_sparse_attn_only = ...
DeepseekV2DecoderLayer.forward_mlp_only = ...
DeepseekV2Model.forward = dmp_forward
```

### 6.5 MoE layer name 处理

文件：`vllm_ascend/ops/fused_moe/fused_moe.py`

位置：`AscendFusedMoE._encode_layer_name()`

```python
def _encode_layer_name(self):
    # DMP forward_context 下返回 Graph/custom-op 可使用的 layer name。
    if ... dmp_context is not None:
        ...

    # 非 DMP 路径保持原有行为。
    return ...
```

### 6.6 KV offload Python 接口

文件：

- `vllm_ascend/kv_offload/asu_npu.py`
- `vllm_ascend/kv_offload/block_location.py`
- `vllm_ascend/kv_offload/kv_loader.py`

位置：`KVLoadOp`、`PlaceholderKVLoadOp`、`SwapBlocksKVLoadOp`、
`BlockLocationTable`、`KVLoader`

```python
class KVLoadOp(...):
    def async_load(...):
        ...

class PlaceholderKVLoadOp(...):
    # 当前 Graph 使用的占位接口。
    ...

class BlockLocationTable:
    def classify_topk_indices(...):
        ...

class KVLoader:
    def async_load_blocks(...):
        ...

    def wait_load_complete(...):
        ...
```

当前 Graph capture 会跳过真正的动态 KV 加载，因此这些接口目前主要用于
eager DMP 和后续真实 KV offload 接入。

### 6.7 环境变量

文件：`vllm_ascend/envs.py`

位置：`env_variables`

```python
env_variables = {
    ...,
    # 控制是否启用 DMP Python 路径。
    "VLLM_ASCEND_ENABLE_DMP": ...,
}
```

### 6.8 单元测试

文件：`tests/ut/worker/test_model_runner_v1.py`

位置：`TestNPUModelRunnerDMP`

```python
class TestNPUModelRunnerDMP(...):
    def test_dmp_graph_eligibility_requires_uniform_decode(self, ...):
        # 验证只有静态、均匀 decode batch 能进入 DMP Graph。
        ...

    def test_dmp_graph_context_is_reused(self):
        # 验证相同 BatchDescriptor 复用同一个 DMPContext。
        ...
```

## 7. 运行时验证标记

成功进入 DMP Graph capture 时，日志应包含：

```text
Prepared DMP full-decode graph context for BatchDescriptor(...)
[DMP] dmp_forward graph_capture=True!
Graph capturing finished ...
```

正式 decode replay 时应包含：

```text
[DMP exec] dmp_context is None=False
Replaying aclgraph
```

Prefill 或 ChunkedPrefill 出现下面的日志属于正常回退：

```text
dmp_context is None=True
```

文本日志可以证明 DMP 路径被 capture 并发生 Graph replay。要进一步证明
单张 NPU 上的两条 Stream 都执行了 kernel，需要在 MindStudio Timeline 中
检查同一个 rank、同一个 replay 区间内的 S0 和 S1。

## 8. 当前限制与后续工作

当前版本已经完成：

- DMP forward 进入 full-decode ACL Graph capture；
- Graph replay 复用 capture 时的 DMP 上下文；
- 单卡 S0/S1 双流 Event 拓扑；
- microbatch A/B metadata 切分；
- A/B Indexer 的同层双流调度；
- Prefill 和不满足静态条件的 batch 安全回退。

当前尚未完成：

- 动态 KV classify/load 进入 Graph；
- 真实 ASU/SSD KV 数据加载；
- A/B MLP 独立执行；
- `MB-A Indexer(L+1)` 与 `MB-B MLP(L)` 的完整跨层重叠；
- DSA-CP 与 DMP Graph 组合；
- NPU profile 中双流重叠比例和性能收益的最终验证。

因此，当前版本应描述为“可 capture/replay 的同层双流 DMP Graph”，不应描述
为已经完成完整跨层 DMP pipeline。
