# Custom AscendC Ops (`op/`)

Standalone NPU operator sources and build scripts for **gather_selection_kv_cache** and **sparse_flash_attention**. Does not import or reference `cann-recipes-infer` at runtime.

## 算子安装与测试

`gather_selection_kv_cache` 与 `sparse_flash_attention` 均可独立编译 OPP + PyTorch `custom_ops` wheel；`lightning_indexer` 示例仍调用 **torch_npu 内置算子**。

### 目录结构

| 路径 | 说明 |
|------|------|
| `ascendc/src/gather_selection_kv_cache/` | Gather AscendC OPP 源码 |
| `ascendc/src/sparse_flash_attention/` | SparseFlashAttention AscendC OPP 源码 |
| `torch_ops_extension/` | PyTorch `custom_ops` wheel（gather / SFA 可分别或合并构建） |
| `examples/` | NPU 精度对比示例 |
| `scripts/` | 编译与测试脚本 |

### 环境要求

- Ascend NPU 容器（已验证：`va-0.18-ess`）
- CANN / ascend-toolkit 已安装
- 首次安装 OPP 后需 source customize vendor 环境

### 一键安装并测试

在容器内执行：

```bash
bash /data/wangpeng/pip-cache/op/scripts/build_and_test_npu.sh
```

依次完成：编译安装 OPP → 编译安装 `custom_ops` wheel → 运行 NPU 示例测试。

### 分步安装

#### gather_selection_kv_cache

**1. 编译并安装 OPP**

```bash
bash /data/wangpeng/pip-cache/op/scripts/build_opp.sh
# 或显式：OPP_OP_NAME=gather_selection_kv_cache bash .../build_opp.sh
```

**2. 编译并安装 PyTorch 扩展**

```bash
bash /data/wangpeng/pip-cache/op/scripts/build_torch_ops.sh
```

#### sparse_flash_attention

**1. 编译并安装 OPP**

```bash
bash /data/wangpeng/pip-cache/op/scripts/build_opp_sfa.sh
```

**2. 编译并安装 PyTorch 扩展**

```bash
bash /data/wangpeng/pip-cache/op/scripts/build_torch_ops_sfa.sh
```

**3. 一键安装并测试**

```bash
bash /data/wangpeng/pip-cache/op/scripts/build_and_test_sfa_npu.sh
```

默认 OPP 安装路径为 `/usr/local/Ascend/cann-8.5.1/opp`，可通过 `INSTALL_OPP` 覆盖。`OPP_OP_NAME` 支持分号分隔多算子，例如 `gather_selection_kv_cache;sparse_flash_attention`。

PyTorch wheel 在临时目录构建，避免挂载目录 git「dubious ownership」问题；产物副本在 `torch_ops_extension/dist/`。

### 运行测试

**统一入口（gather 算子）**

```bash
bash /data/wangpeng/pip-cache/op/scripts/run_npu_tests.sh
```

脚本会自动配置 CANN / customize 环境与 `LD_LIBRARY_PATH`，并运行 `examples/test_npu_gather_selection_kv_cache.py` 中的 `TestCustomGatherSelectionKvCache`（eager / graph / quant graph / TND graph，共 4 个用例）。

**sparse_flash_attention（custom OPP + wheel）**

```bash
bash /data/wangpeng/pip-cache/op/scripts/run_npu_tests_sfa.sh
```

**lightning_indexer / sparse_flash_attention（torch_npu 内置，eager）**

```bash
cd /data/wangpeng/pip-cache/op/examples

# lightning_indexer：BSND + TND，共 2 个 eager 用例
python3 -c "
import sys
from test_npu_lightning_indexer import _parse_test_names, _run_tests
_run_tests(_parse_test_names(sys.argv))
"

# sparse_flash_attention（内置接口）：BSND + TND/PA_BSND
python3 -c "
import sys
from test_npu_sparse_flash_attention import _parse_test_names, _run_tests
_run_tests(_parse_test_names(sys.argv))
"
```

> 建议通过 `python3 -c` 以模块方式导入测试，避免直接 `python3 test_*.py` 在部分 NPU 环境下触发 segfault。

### 运行时环境（手动调试）

每个新 shell 需加载：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u && source /usr/local/Ascend/cann-8.5.1/opp/vendors/customize/bin/set_env.bash && set -u

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/customize/op_api/lib/:\
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/lib:\
/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:\
${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0
```

验证 gather 扩展：`python3 -c "import custom_ops"`（请在 `/tmp` 等非源码目录下执行，勿在 `torch_ops_extension/` 内 import）。
