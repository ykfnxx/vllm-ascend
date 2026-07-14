"""vllm-ascend 侧维护的 DSA 稀疏卸载实现包。

这个包承载 DeepSeek-V3.2 DSA 稀疏卸载在昇腾后端上的主要实现，包括
请求/缓存元数据、Indexer/MLA cache 解耦后的资源管理、DRAM 热层存储、
lookup-resident 算子语义对接、图模式 gate，以及迁移出 vLLM 主仓后的
运行时 patch 辅助代码。这里的模块都应以 vllm-ascend 为 owner，避免
再把 DSA/Ascend 专有逻辑扩散回 vLLM 项目。
"""
