# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 稀疏卸载使用的 KV-cache spec 语义判定工具。

DSA 通过 vllm-ascend 的运行时 patch 接入，而部分 KV-cache spec 类型来自
vLLM 原生模块或 vllm-ascend patch 后的模块。这里统一用“Indexer dense
plane / MLA resident plane”等语义判断替代散落的 class 名称判断，减少
后续迁移或上游字段变化时的适配面。
"""

from __future__ import annotations

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    IndexerKVSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)


def is_dsa_indexer_spec(spec: KVCacheSpec) -> bool:
    """Return whether ``spec`` is the dense indexer-cache plane."""
    return isinstance(spec, IndexerKVSpec)


def is_dsa_mla_resident_spec(spec: KVCacheSpec) -> bool:
    """Return whether ``spec`` is the MLA/full resident-cache plane."""
    if is_dsa_indexer_spec(spec):
        return False
    return isinstance(spec, (MLAAttentionSpec, FullAttentionSpec))
