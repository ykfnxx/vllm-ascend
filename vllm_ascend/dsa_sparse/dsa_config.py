# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 稀疏卸载的拉起参数解析与兼容入口。

本文件把用户在 ``additional_config["dsa_sparse_config"]`` 中传入的字段
规范化到 vLLM/vLLM-Ascend 运行时真正读取的配置位置。它只做配置解析、
默认值补齐和字段校验，不参与推理过程中的动态状态更新；后续新增 DSA
拉起参数时，优先在这里集中声明映射关系，避免配置入口再次散落到多个文件。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm_ascend.dsa_sparse.dsa_graph_gate import (
    DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY,
)
from vllm_ascend.dsa_sparse.dsa_trace import DSA_TRACE_CONFIG_KEY

DSA_SPARSE_ADDITIONAL_CONFIG_KEY = "dsa_sparse_config"
DSA_SPARSE_SUPPORTED_ARCHITECTURES = frozenset({
    "DeepseekV32ForCausalLM",
    "GlmMoeDsaForCausalLM",
})
_DSA_GRAPH_PUBLIC_CONFIG_KEY = "enable_row_mode_decode_graph"

_DSA_SPARSE_CONFIG_FIELD_MAPPINGS = (
    ("enabled", "enable_dsa_sparse_cache"),
    ("split_indexer_cache", "enable_dsa_split_indexer_cache"),
    ("indexer_mla_block_ratio", "dsa_indexer_mla_block_ratio"),
    ("hbm_sparse_budget", "dsa_hbm_sparse_budget"),
    ("hbm_resident_tokens", "dsa_hbm_resident_tokens"),
    ("max_active_reqs", "dsa_max_active_reqs"),
    ("hot_cpu_block_multiple", "dsa_hot_cpu_block_multiple"),
)
_DSA_SPARSE_DEFAULT_CACHE_ATTRS: dict[str, Any] = {
    "enable_dsa_sparse_cache": False,
    "enable_dsa_split_indexer_cache": False,
    "dsa_indexer_mla_block_ratio": 3,
    "dsa_hbm_sparse_budget": 2048,
    "dsa_hbm_resident_tokens": 8192,
    # Direct token->slot tables are per request and per layer. Deployments must
    # size this existing request-pool limit together with index HBM usage.
    "dsa_max_active_reqs": 256,
    "dsa_hot_cpu_block_multiple": 3,
}
_DSA_SPARSE_PUBLIC_KEYS = frozenset(
    {public for public, _ in _DSA_SPARSE_CONFIG_FIELD_MAPPINGS}
    | {_DSA_GRAPH_PUBLIC_CONFIG_KEY, "trace_points"}
)


def _normalize_dsa_trace_points_config(trace_config: Any) -> dict[str, Any]:
    if isinstance(trace_config, bool):
        return {"enabled": trace_config}
    if not isinstance(trace_config, dict):
        raise TypeError(
            "dsa_sparse_config['trace_points'] must be a dict or bool, got "
            f"{type(trace_config)!r}")

    supported_keys = frozenset({"enabled", "points", "ranks", "layers", "sync"})
    unknown = sorted(set(trace_config) - supported_keys)
    if unknown:
        raise ValueError(
            "Unknown dsa_sparse_config['trace_points'] key(s): "
            f"{', '.join(unknown)}. Supported keys: {sorted(supported_keys)}")

    def normalize_sequence(value: Any) -> Any:
        if value is None or isinstance(value, (str, bytes, bytearray)):
            return value
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        if isinstance(value, Sequence):
            return list(value)
        return value

    return {
        key: normalize_sequence(value)
        for key, value in trace_config.items()
    }


def _normalize_dsa_sparse_config(
    raw_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    unknown = sorted(set(raw_config) - _DSA_SPARSE_PUBLIC_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown dsa_sparse_config key(s): {', '.join(unknown)}. "
            f"Supported keys: {sorted(_DSA_SPARSE_PUBLIC_KEYS)}")

    cache_attrs = dict(_DSA_SPARSE_DEFAULT_CACHE_ATTRS)
    for public_name, cache_attr in _DSA_SPARSE_CONFIG_FIELD_MAPPINGS:
        if public_name in raw_config:
            cache_attrs[cache_attr] = raw_config[public_name]

    if cache_attrs["enable_dsa_sparse_cache"]:
        if cache_attrs["enable_dsa_split_indexer_cache"] is False and (
                "split_indexer_cache" in raw_config):
            raise ValueError(
                "dsa_sparse_config['enabled']=True requires "
                "dsa_sparse_config['split_indexer_cache']=True")
        cache_attrs["enable_dsa_split_indexer_cache"] = True
        sparse_topk = int(cache_attrs["dsa_hbm_sparse_budget"])
        resident_tokens = int(cache_attrs["dsa_hbm_resident_tokens"])
        if sparse_topk <= 0:
            raise ValueError(
                "dsa_sparse_config['hbm_sparse_budget'] must be positive")
        if resident_tokens <= sparse_topk:
            raise ValueError(
                "dsa_sparse_config['hbm_resident_tokens'] must be greater "
                "than hbm_sparse_budget")

    additional_updates: dict[str, Any] = {}
    if _DSA_GRAPH_PUBLIC_CONFIG_KEY in raw_config:
        if bool(raw_config[_DSA_GRAPH_PUBLIC_CONFIG_KEY]):
            raise ValueError(
                "DSA lookup resident cache does not support row-mode decode "
                "graph until lookup/miss materialization/maintenance are "
                "provided as capture-safe NPU operators")
        additional_updates[DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY] = (
            raw_config[_DSA_GRAPH_PUBLIC_CONFIG_KEY])
    if "trace_points" in raw_config:
        additional_updates[DSA_TRACE_CONFIG_KEY] = (
            _normalize_dsa_trace_points_config(raw_config["trace_points"]))

    return cache_attrs, additional_updates


def attach_dsa_sparse_cache_attrs(vllm_config: Any) -> None:
    """Attach DSA cache knobs from ``additional_config`` onto CacheConfig.

    vLLM's core ``CacheConfig`` is backend-agnostic. Users pass DSA
    sparse-offload settings through ``additional_config["dsa_sparse_config"]``;
    vllm-ascend then materializes them as dynamic cache attributes before its
    platform checks and KV-cache allocation patches read those knobs.
    """
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return

    cache_attrs = additional_config.get(DSA_SPARSE_ADDITIONAL_CONFIG_KEY)
    if cache_attrs is None:
        return
    if not isinstance(cache_attrs, dict):
        raise TypeError(
            f"additional_config[{DSA_SPARSE_ADDITIONAL_CONFIG_KEY!r}] must "
            f"be a dict, got {type(cache_attrs)!r}")

    merged_attrs, additional_updates = _normalize_dsa_sparse_config(cache_attrs)
    if (merged_attrs["enable_dsa_sparse_cache"]
            and additional_config.get(
                DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY) is True):
        raise ValueError(
            "DSA lookup resident cache does not support row-mode decode "
            "graph until lookup/miss materialization/maintenance are "
            "provided as capture-safe NPU operators")
    for key, value in additional_updates.items():
        if key in additional_config and additional_config[key] != value:
            raise ValueError(
                "Conflicting DSA sparse-offload config for "
                f"additional_config[{key!r}]: {additional_config[key]!r} "
                f"vs {value!r}")
        additional_config[key] = value

    if bool(additional_updates.get(
            DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY, False)):
        ascend_compile_config = additional_config.setdefault(
            "ascend_compilation_config", {})
        if not isinstance(ascend_compile_config, dict):
            raise TypeError(
                "additional_config['ascend_compilation_config'] must be a "
                f"dict when DSA graph is enabled, got "
                f"{type(ascend_compile_config)!r}")
        if ascend_compile_config.get("enable_npugraph_ex", False):
            raise ValueError(
                "DSA row-mode decode graph does not support "
                "ascend_compilation_config['enable_npugraph_ex']=True. "
                "DSA graph capture only targets row-mode decode; npugraph_ex "
                "also compiles profile/prefill paths and can fail inside "
                "MoE communication operators. Please set it to False or omit "
                "it.")
        # vllm-ascend defaults npugraph_ex to True. DSA graph mode needs the
        # ACL full-graph replay path, but does not want TorchAir to compile the
        # profiling prefill path before KV-cache split metadata is initialized.
        ascend_compile_config["enable_npugraph_ex"] = False

    for key, value in merged_attrs.items():
        object.__setattr__(vllm_config.cache_config, key, value)


def is_dsa_sparse_config_enabled(vllm_config: Any) -> bool:
    """Return whether DSA sparse offload is requested by user config.

    Some call sites run across vLLM multiprocessing/pydantic boundaries where
    dynamic CacheConfig attributes may not have been materialized yet. Treat
    ``additional_config["dsa_sparse_config"].enabled`` as the source of truth,
    while still accepting an already-attached cache flag.
    """
    if vllm_config is None:
        return False

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None and bool(
            getattr(cache_config, "enable_dsa_sparse_cache", False)):
        return True

    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return False
    dsa_config = additional_config.get(DSA_SPARSE_ADDITIONAL_CONFIG_KEY)
    return isinstance(dsa_config, dict) and bool(dsa_config.get("enabled"))
