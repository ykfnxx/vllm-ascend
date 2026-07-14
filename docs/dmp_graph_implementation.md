# DMP Graph 修改说明

本次复用 vLLM-Ascend 现有的 `FULL_DECODE_ONLY` ACL Graph，只修改 Python
层，使 DMP 双流路径在 Graph capture 时生效。

## 修改位置

### `vllm_ascend/ascend_forward_context.py`

- `set_ascend_forward_context()`：增加 `dmp_context` 参数并写入
  `forward_context`。

### `vllm_ascend/worker/model_runner_v1.py`

- `NPUModelRunner.__init__()`：缓存 DMP Stream 和 Graph Context。
- `_is_dmp_eligible()`：限制当前只处理规则的 DecodeOnly batch。
- `_maybe_create_dmp_slices()`：切分 A/B microbatch 及 attention metadata。
- `_get_or_create_dmp_graph_context()`：按 batch shape 创建并缓存 DMPContext。
- `_dummy_run()`：Graph capture 时提前进入 DMP forward。
- `execute_model()`：Graph replay 时复用 capture 阶段的 DMPContext。

```python
class NPUModelRunner(...):
    def _get_or_create_dmp_graph_context(...):
        # 创建并缓存 DMPContext、Stream 和 Event。
        ...

    def _dummy_run(...):
        # capture 时把 DMPContext 传入模型 forward。
        ...

    def execute_model(...):
        # replay 时取得同一个 DMPContext。
        ...
```

### `vllm_ascend/worker/dmp_context.py`

- `DMPSlice`：记录 A/B microbatch 的 token 范围。
- `DMPContext.get_event()`：缓存跨流 Event。
- `prepare_graph_events()`：在 capture 前创建 Event。
- `enter_microbatch()`：切换当前 microbatch 的 metadata。
- `slice_hidden_states()` / `merge_hidden_states()`：切分和合并输出。

### `vllm_ascend/patch/worker/patch_deepseek_mtp.py`

- `forward_indexer_only()`：单独执行 Indexer。
- `forward_sparse_attn_only()`：单独执行 Sparse Attention。
- `forward_mlp_only()`：单独执行 MLP。
- `forward_mlp_two_mb_once()`：当前合并执行 A/B MLP。
- `dmp_forward()`：实现 S0/S1 调度及 Event 依赖。
- 设置 `IGNORE_COMPILE_KEY`：跳过 TorchDynamo 对双流 Python 控制流的追踪。

```python
def dmp_forward(...):
    # S0：ACL Graph 主 Stream。
    # S1：DMP Stream。
    ...

    # A Indexer 放在 S1，B Indexer 和后续计算放在 S0。
    ...

    # capture 时跳过动态 KV classify/load 和 Host IO。
    ...
```

### `vllm_ascend/attention/sfa_v1.py`

- `AscendSFAImpl.forward_indexer_only()`：拆出 Indexer 阶段。
- `AscendSFAImpl.forward_sparse_attn_only()`：拆出 Sparse Attention 阶段。

### `vllm_ascend/ops/mla.py`

- `AscendMLA.forward_indexer_only()` / `forward_sparse_attn_only()`：调用拆分后的
  MLA 路径。
- `mla_forward_indexer_only()` / `mla_forward_sparse_attn_only()`：增加对应的
  Python custom-op 包装和注册。

### 其他文件

- `vllm_ascend/ops/fused_moe/fused_moe.py`
  - `_encode_layer_name()`：兼容 DMP forward context。
- `vllm_ascend/kv_offload/asu_npu.py`
  - 增加 KV load 接口和当前使用的占位实现。
- `vllm_ascend/kv_offload/block_location.py`
  - 增加 KV block 位置记录和分类。
- `vllm_ascend/kv_offload/kv_loader.py`
  - 增加异步加载及 Event 管理。
- `vllm_ascend/envs.py`
  - 增加 `VLLM_ASCEND_ENABLE_DMP` 开关。
- `tests/ut/worker/test_model_runner_v1.py`
  - 增加 DMP Graph 条件和 Context 复用测试。

当前完成的是同层双流 Graph；动态 KV 加载和完整跨层 DMP 流水尚未加入。
