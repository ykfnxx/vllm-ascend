"""DSA 稀疏卸载的可选 trace 打点配置。

本文件定义 ``dsa_sparse_config.trace_points`` 的解析和查询逻辑，用于在
调试时打开 lightning-indexer、gather-selection 等关键边界的少量日志。
trace 配置在拉起时解析，推理路径只做只读查询；默认关闭，避免性能测试时
引入 host 日志或同步开销。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

DSA_TRACE_CONFIG_KEY = "dsa_sparse_trace_points"
DSA_TRACE_POINT_LIGHTNING_INDEXER = "lightning_indexer"
DSA_TRACE_POINT_GATHER_SELECTION = "gather_selection"
DSA_TRACE_POINT_GATHER_SELECTION_STATS = "gather_selection_stats"
DSA_TRACE_ALL_POINTS = frozenset({
    DSA_TRACE_POINT_LIGHTNING_INDEXER,
    DSA_TRACE_POINT_GATHER_SELECTION,
    DSA_TRACE_POINT_GATHER_SELECTION_STATS,
})


@dataclass(frozen=True)
class DSATraceConfig:
    enabled: bool = False
    points: frozenset[str] = frozenset()
    ranks: frozenset[int] | None = None
    layers: frozenset[int] | None = frozenset({0})
    sync: bool = False


_DSA_TRACE_CONFIG = DSATraceConfig()


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _parse_csv_or_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _parse_int_filter(value: Any) -> frozenset[int] | None:
    if value is None or value in ("*", "all"):
        return None
    parsed: set[int] = set()
    for item in _parse_csv_or_iterable(value):
        if isinstance(item, str) and item.strip().lower() in ("*", "all"):
            return None
        parsed.add(int(item))
    return frozenset(parsed)


def _parse_points(value: Any) -> frozenset[str]:
    if value is None or value in ("*", "all"):
        return DSA_TRACE_ALL_POINTS
    items = [str(item).strip() for item in _parse_csv_or_iterable(value)]
    if any(item.lower() in ("*", "all") for item in items):
        return DSA_TRACE_ALL_POINTS
    points = {item for item in items if item}
    if not points:
        return frozenset()
    unknown = sorted(points - DSA_TRACE_ALL_POINTS)
    if unknown:
        raise ValueError(
            f"Unknown DSA trace point(s): {unknown}. Supported points: "
            f"{sorted(DSA_TRACE_ALL_POINTS)}")
    return frozenset(points)


def configure_dsa_trace(trace_config: Any) -> DSATraceConfig:
    """Parse public DSA trace config once at model-runner initialization."""
    global _DSA_TRACE_CONFIG

    if trace_config is None:
        _DSA_TRACE_CONFIG = DSATraceConfig()
        return _DSA_TRACE_CONFIG

    if isinstance(trace_config, bool):
        config = {"enabled": trace_config}
    elif isinstance(trace_config, dict):
        config = dict(trace_config)
    else:
        raise TypeError(
            "DSA trace config must be a dict or bool, got "
            f"{type(trace_config)!r}")

    enabled = _as_bool(config.get("enabled"), default=False)
    if not enabled:
        _DSA_TRACE_CONFIG = DSATraceConfig()
        return _DSA_TRACE_CONFIG

    _DSA_TRACE_CONFIG = DSATraceConfig(
        enabled=True,
        points=_parse_points(config.get("points")),
        ranks=_parse_int_filter(config.get("ranks")),
        layers=_parse_int_filter(config.get("layers", [0])),
        sync=_as_bool(config.get("sync"), default=False),
    )
    logger.info("Configured DSA trace points: %s", _DSA_TRACE_CONFIG)
    return _DSA_TRACE_CONFIG


def _layer_id_from_name(layer_name: str | None) -> int | None:
    if layer_name is None:
        return None
    try:
        return int(str(layer_name).split(".")[2])
    except (IndexError, TypeError, ValueError):
        return None


def _current_tp_rank() -> int | None:
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return None


def dsa_trace_enabled(
    point: str,
    *,
    layer_name: str | None = None,
    layer_id: int | None = None,
    tp_rank: int | None = None,
) -> bool:
    config = _DSA_TRACE_CONFIG
    if not config.enabled or point not in config.points:
        return False
    if config.ranks is not None:
        if tp_rank is None:
            tp_rank = _current_tp_rank()
        if tp_rank is None:
            return False
        if int(tp_rank) not in config.ranks:
            return False
    if config.layers is not None:
        resolved_layer_id = layer_id
        if resolved_layer_id is None:
            resolved_layer_id = _layer_id_from_name(layer_name)
        if (resolved_layer_id is None
                or int(resolved_layer_id) not in config.layers):
            return False
    return True


def dsa_trace_sync_enabled(point: str) -> bool:
    config = _DSA_TRACE_CONFIG
    return bool(config.enabled and config.sync and point in config.points)
