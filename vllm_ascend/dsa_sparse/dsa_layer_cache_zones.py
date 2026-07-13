"""DSA layer-wise cache zone discovery and registry.

本文件只负责 layer 维度的 cache tensor 绑定与解析：从 ForwardContext 中找到
当前 attention layer 对应的 MLA noPE/ROPE cache、indexer cache，并在 worker
生命周期内校验这些 cache zone 不发生漂移。

这里不承载 request 跨 step 状态，也不构造单个 model forward 的 batch 张量。
request-row 元数据留在 dsa_req_meta.py；forward-level tensor 化逻辑留在
dsa_forward_batch.py。
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.forward_context import ForwardContext


@dataclass(frozen=True)
class LayerCacheZones:
    nopek_cache_zone: object
    ropek_cache_zone: object
    indexer_cache_zone: object | None
    layerwise_global_block_size: int | None


class DSALayerCacheRegistry:
    """Persistent cache-zone registry for one DSA worker instance.

    KV cache tensors are allocated during worker/model-runner initialization
    and are expected to stay stable for the worker lifetime. DSA keeps these
    layer zones here so begin/finish hooks can use a small layer id lookup
    instead of re-discovering and overwriting cache bindings every layer call.

    If a later forward sees different cache tensors for the same layer, that is
    treated as a lifecycle violation and reported explicitly. The intended
    recovery path for a true KV cache rebuild is to recreate/reinitialize the
    DSA worker-side manager, not to silently reuse stale residency metadata.
    """

    def __init__(self, num_layers: int | None = None) -> None:
        # Layer ids are dense model layer indices, so a list gives the hot
        # attention_begin path one direct index operation instead of a dict
        # lookup. The list can still grow in tests or unusual initialization
        # paths where the final layer count is not known yet.
        initial_layers = 0 if num_layers is None else max(0, int(num_layers))
        self._cache_zones_by_layer: list[LayerCacheZones | None] = [
            None for _ in range(initial_layers)
        ]

    def _ensure_layer_capacity(self, layer_id: int) -> None:
        if layer_id < 0:
            raise RuntimeError(f"DSA got invalid negative layer id {layer_id}")
        missing = layer_id + 1 - len(self._cache_zones_by_layer)
        if missing > 0:
            self._cache_zones_by_layer.extend([None] * missing)

    @staticmethod
    def _same_cache_object(left: object, right: object) -> bool:
        if left is right:
            return True
        left_data_ptr = getattr(left, "data_ptr", None)
        right_data_ptr = getattr(right, "data_ptr", None)
        if callable(left_data_ptr) and callable(right_data_ptr):
            try:
                return (
                    left_data_ptr() == right_data_ptr()
                    and getattr(left, "shape", None)
                    == getattr(right, "shape", None)
                    and getattr(left, "dtype", None)
                    == getattr(right, "dtype", None)
                    and getattr(left, "device", None)
                    == getattr(right, "device", None)
                )
            except Exception:
                return False
        return False

    @classmethod
    def _same_cache_zones(cls, left: LayerCacheZones,
                          right: LayerCacheZones) -> bool:
        return (
            cls._same_cache_object(left.nopek_cache_zone,
                                   right.nopek_cache_zone)
            and cls._same_cache_object(left.ropek_cache_zone,
                                       right.ropek_cache_zone)
            and cls._same_cache_object(left.indexer_cache_zone,
                                       right.indexer_cache_zone)
            and left.layerwise_global_block_size
            == right.layerwise_global_block_size
        )

    def bind_or_validate(self, layer_id: int,
                         cache_zones: LayerCacheZones) -> LayerCacheZones:
        """Bind a layer once, then verify later observations are identical."""
        layer_id = int(layer_id)
        self._ensure_layer_capacity(layer_id)
        existing = self._cache_zones_by_layer[layer_id]
        if existing is None:
            self._cache_zones_by_layer[layer_id] = cache_zones
            return cache_zones
        if not self._same_cache_zones(existing, cache_zones):
            raise RuntimeError(
                f"DSA layer cache zones changed for layer {layer_id}; "
                "KV cache tensors must stay stable for one worker lifetime")
        return existing

    def get(self, layer_id: int) -> LayerCacheZones | None:
        layer_id = int(layer_id)
        if layer_id < 0 or layer_id >= len(self._cache_zones_by_layer):
            return None
        return self._cache_zones_by_layer[layer_id]

    def require(self, layer_id: int) -> LayerCacheZones:
        cache_zones = self.get(layer_id)
        if cache_zones is None:
            raise RuntimeError(
                f"DSA layer cache registry has no zones for layer {layer_id}")
        return cache_zones


def resolve_layer_cache_zones(
        layer_name: str,
        forward_context: ForwardContext,
) -> LayerCacheZones:
    attn = forward_context.no_compile_layers[layer_name]
    virtual_engine = forward_context.virtual_engine
    sfa_cache = attn.mla_attn.kv_cache[virtual_engine]
    if not isinstance(sfa_cache, (tuple, list)) or len(sfa_cache) < 2:
        raise RuntimeError(
            f"DSA requires MLA cache zones for {layer_name}, got "
            f"{type(sfa_cache).__name__}")
    nopek_cache_zone = sfa_cache[0]
    ropek_cache_zone = sfa_cache[1]

    indexer_layer_name = attn.mla_attn.impl.indexer_k_cache_layer_name
    if indexer_layer_name is None:
        raise RuntimeError(
            f"DSA requires a split indexer cache layer for {layer_name}")
    indexer_layer = forward_context.no_compile_layers[indexer_layer_name]
    indexer_cache_zone = indexer_layer.kv_cache[virtual_engine]
    if not hasattr(indexer_cache_zone, "shape"):
        raise RuntimeError(
            f"DSA requires a tensor indexer cache for {indexer_layer_name}, "
            f"got {type(indexer_cache_zone).__name__}")

    shape = getattr(nopek_cache_zone, "shape", None)
    layerwise_global_block_size = (
        int(shape[0]) if shape is not None and len(shape) > 0 else None)
    return LayerCacheZones(
        nopek_cache_zone=nopek_cache_zone,
        ropek_cache_zone=ropek_cache_zone,
        indexer_cache_zone=indexer_cache_zone,
        layerwise_global_block_size=layerwise_global_block_size,
    )
