# A5 MTP KV Gather Sim 图模式同步包

本包把源码同步到 `/root/dmp/vllm-ascend-0.23.0`，然后执行固定的
`batch=64、prompt=131072、output=10` P/D 测试。Prefill 使用 Eager，Decode
启用 MTP（1 个 speculative token）和 `FULL_DECODE_ONLY` 图，且只采集 Decode
Profile。DMP 明确关闭。

P/D KV 传输使用当前分支的 `LocalShmConnector`。

Decode 路径使用：

- `dsa_offload_lookup_update_batch`；
- 原生 `AsuKvGather` 模拟算子；
- 合成的全零远端 KV/RoPE payload。

同步过程不会下载依赖，也不会删除目标目录中的额外文件。被覆盖的旧文件保存在
`/root/dmp/backups/`。由于新增了 C++ 绑定和自定义算子，A5 上第一次运行需要完成
一次原生编译；源码指纹不变时，后续运行会直接复用已安装产物。如果上一次已经
安装成功、但在 smoke 检查阶段中断而尚未写入指纹，本包会先验证现有算子并补写
指纹，不会重复执行 pip/CMake 编译。运行前会把仓库内生成的
`libcust_opapi.so` 同时加入 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`，并
验证 `aclnnAsuKvGather` 两个入口符号后再调用算子。

如果保留的增量构建目录让新增算子的单算子 Kernel 已生成、但总索引仍是旧版本，
本包会从已安装的单算子 JSON 原子重建 `binary_info_config.json` 和
`relocatable_kernel_info_config.json`。旧索引会备份到本次运行结果目录；该修复
不调用 pip/CMake，也不会重新编译 Kernel。

```bash
unzip dsa-offload-0.23-graph-a5.zip
cd dsa-offload-0.23-graph-a5
MODEL_PATH=/root/dmp/models/glm-5.2-4layer \
A5_PREFILL_DEVICE=3 \
A5_DECODE_DEVICE=5 \
A5_RECREATE_CONTAINER=0 \
bash apply_and_run_on_host.sh
```

如果该专用容器已存在，但挂载或 NPU 选择与本次不同，只需第一次把
`A5_RECREATE_CONTAINER` 改成 `1`。模型位于其他目录时修改 `MODEL_PATH`。

成功标记为：

```text
A5_DSA_OFFLOAD_MTP_KVGATHER_SIM_64X128K_GRAPH_PASSED
```

结果目录位于 `/root/dmp/vllm-ascend-0.23.0/scripts/results/<timestamp>/`，其中
`profile/` 下的 `*_ascend_pt` 目录可以直接导入 MindStudio Insight。
