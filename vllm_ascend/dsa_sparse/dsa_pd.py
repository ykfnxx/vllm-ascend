"""Serializable control-plane metadata for DSA KVIO P/D handoff."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from vllm_ascend.dsa_sparse.dsa_resident_pool import (
    DSA_LOOKUP_INDEX_CAPACITY,
    DSA_LOOKUP_QUERY_TOKENS,
    DSA_LOOKUP_RESIDENT_TOKENS,
)

DSA_KVIO_CONNECTOR_NAME = "DSAKVIOConnector"
DSA_MOONCAKE_CONNECTOR_NAME = "DSAMooncakeConnector"
DSA_KVIO_PD_MANIFEST_KEY = "dsa_kvio_manifest"
DSA_KVIO_PD_LAYER_TOPK_KEY = "dsa_kvio_layer_topk_by_rank"
DSA_PD_INITIAL_TRANSPORT_KEY = "dsa_initial_transport"
DSA_PD_INITIAL_TRANSPORT_KVIO = "kvio"
DSA_PD_INITIAL_TRANSPORT_MOONCAKE = "mooncake"
DSA_KVIO_PD_PROTOCOL_VERSION = 3
DSA_KVIO_PD_STATE_READY = 1


def _config_value(config: Any, name: str) -> Any:
    value = getattr(config, name, None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_dsa_kvio_layout_fingerprint(vllm_config: Any) -> int:
    """Build a stable P/D fingerprint for the KVIO storage/cache layout."""
    cache_config = getattr(vllm_config, "cache_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    get_num_layers = getattr(
        model_config, "get_total_num_hidden_layers", None)
    try:
        num_layers = (
            int(get_num_layers()) if callable(get_num_layers) else None)
    except (TypeError, ValueError):
        num_layers = None
    if num_layers is None:
        num_layers = _config_value(text_config, "num_hidden_layers")

    cache_layout_fields = (
        "head_dim",
        "hidden_size",
        "index_head_dim",
        "index_topk",
        "kv_lora_rank",
        "num_attention_heads",
        "num_key_value_heads",
        "qk_rope_head_dim",
    )
    payload = {
        "block_size": _config_value(cache_config, "block_size"),
        "cache_dtype": _config_value(cache_config, "cache_dtype"),
        "index_capacity": DSA_LOOKUP_INDEX_CAPACITY,
        "max_model_len": _config_value(model_config, "max_model_len"),
        "model": _config_value(model_config, "model"),
        "model_dtype": _config_value(model_config, "dtype"),
        "model_id": _config_value(cache_config, "dsa_kvio_model_id"),
        "num_layers": num_layers,
        "quantization": _config_value(model_config, "quantization"),
        "pipeline_parallel_size": _config_value(
            parallel_config, "pipeline_parallel_size"),
        "query_tokens": DSA_LOOKUP_QUERY_TOKENS,
        "resident_tokens": DSA_LOOKUP_RESIDENT_TOKENS,
        "revision": _config_value(model_config, "revision"),
        "tensor_parallel_size": _config_value(
            parallel_config, "tensor_parallel_size"),
        "text_cache_layout": {
            field_name: _config_value(text_config, field_name)
            for field_name in cache_layout_fields
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.blake2b(encoded, digest_size=8).digest()
    return (
        int.from_bytes(digest, byteorder="little")
        & 0x7FFFFFFFFFFFFFFF
    )


@dataclass(frozen=True)
class DSAKVIOPDManifest:
    """Compact P-built plan used to initialize one D-side DSA request.

    Physical HBM block ids are deliberately absent: they are allocated by the
    D scheduler.  The manifest describes the logical KVIO layout and the
    resident/tail layout that the D worker must materialize after allocation.
    Per-layer TopK seeds are carried beside this manifest in route params.
    """

    protocol_version: int
    state: int
    generation: int
    remote_request_id: int
    model_id: int
    producer_world_size: int
    layout_fingerprint: int
    stored_token_count: int
    block_size: int
    logical_block_count: int
    index_capacity: int
    resident_tokens: int
    free_slot_tokens: int
    resident_slot_start: int
    resident_token_count: int
    tail_token_start: int
    tail_token_count: int
    tail_slot_start: int

    @classmethod
    def build(
        cls,
        *,
        remote_request_id: int,
        model_id: int,
        stored_token_count: int,
        block_size: int,
        index_capacity: int,
        resident_tokens: int,
        free_slot_tokens: int,
        producer_world_size: int,
        layout_fingerprint: int,
        generation: int | None = None,
    ) -> "DSAKVIOPDManifest":
        stored_token_count = int(stored_token_count)
        block_size = int(block_size)
        if stored_token_count <= 0:
            raise ValueError(
                "DSA KVIO P/D manifest requires stored_token_count > 0")
        if block_size <= 0:
            raise ValueError("DSA KVIO P/D manifest requires block_size > 0")

        resident_token_count = min(int(resident_tokens), stored_token_count)
        tail_token_start = (stored_token_count // block_size) * block_size
        tail_token_count = stored_token_count - tail_token_start
        if generation is None:
            generation = int(remote_request_id)
        manifest = cls(
            protocol_version=DSA_KVIO_PD_PROTOCOL_VERSION,
            state=DSA_KVIO_PD_STATE_READY,
            generation=int(generation),
            remote_request_id=int(remote_request_id),
            model_id=int(model_id),
            producer_world_size=int(producer_world_size),
            layout_fingerprint=int(layout_fingerprint),
            stored_token_count=stored_token_count,
            block_size=block_size,
            logical_block_count=(stored_token_count + block_size - 1)
            // block_size,
            index_capacity=int(index_capacity),
            resident_tokens=int(resident_tokens),
            free_slot_tokens=int(free_slot_tokens),
            resident_slot_start=0,
            resident_token_count=resident_token_count,
            tail_token_start=tail_token_start,
            tail_token_count=tail_token_count,
            tail_slot_start=int(resident_tokens) + int(free_slot_tokens),
        )
        manifest.validate()
        return manifest

    def to_dict(self) -> dict[str, int]:
        return {
            field_name: int(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DSAKVIOPDManifest":
        if not isinstance(raw, dict):
            raise TypeError(
                "DSA KVIO P/D manifest must be a dict, got "
                f"{type(raw)!r}")
        missing = sorted(set(cls.__dataclass_fields__) - set(raw))
        if missing:
            raise ValueError(
                "DSA KVIO P/D manifest is missing field(s): "
                f"{', '.join(missing)}")
        manifest = cls(**{
            field_name: int(raw[field_name])
            for field_name in cls.__dataclass_fields__
        })
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.protocol_version != DSA_KVIO_PD_PROTOCOL_VERSION:
            raise ValueError(
                "Unsupported DSA KVIO P/D manifest version: "
                f"{self.protocol_version}")
        if self.state != DSA_KVIO_PD_STATE_READY:
            raise ValueError(
                "DSA KVIO P/D manifest is not ready: "
                f"state={self.state}")
        if self.generation < 0:
            raise ValueError(
                "DSA KVIO P/D manifest has a negative generation")
        if self.producer_world_size <= 0:
            raise ValueError(
                "DSA KVIO P/D manifest has an invalid producer world size")
        if self.layout_fingerprint <= 0:
            raise ValueError(
                "DSA KVIO P/D manifest has an invalid layout fingerprint")
        if self.stored_token_count <= 0 or self.block_size <= 0:
            raise ValueError(
                "DSA KVIO P/D manifest has invalid token/block counts")
        if (
            self.index_capacity <= 0
            or self.resident_tokens <= 0
            or self.free_slot_tokens <= 0
        ):
            raise ValueError(
                "DSA KVIO P/D manifest has invalid lookup capacities")
        if self.resident_tokens + self.free_slot_tokens > self.index_capacity:
            raise ValueError(
                "DSA KVIO P/D resident layout exceeds lookup index capacity")
        if self.remote_request_id < 0 or self.model_id < 0:
            raise ValueError("DSA KVIO P/D manifest has a negative id")
        if self.stored_token_count > self.index_capacity:
            raise ValueError(
                "DSA KVIO P/D stored tokens exceed lookup index capacity")
        expected_blocks = (
            self.stored_token_count + self.block_size - 1
        ) // self.block_size
        if self.logical_block_count != expected_blocks:
            raise ValueError(
                "DSA KVIO P/D manifest logical block count mismatch: "
                f"{self.logical_block_count} vs {expected_blocks}")
        if self.resident_slot_start != 0:
            raise ValueError(
                "DSA KVIO P/D initial resident slots must start at zero")
        expected_resident_tokens = min(
            self.resident_tokens, self.stored_token_count)
        if self.resident_token_count != expected_resident_tokens:
            raise ValueError(
                "DSA KVIO P/D initial resident token count is invalid")
        expected_tail_start = (
            self.stored_token_count // self.block_size
        ) * self.block_size
        if (
            self.tail_token_start != expected_tail_start
            or self.tail_token_count
            != self.stored_token_count - expected_tail_start
        ):
            raise ValueError("DSA KVIO P/D tail range is inconsistent")
        if self.tail_token_start < self.resident_token_count:
            raise ValueError(
                "DSA KVIO P/D history before the dense tail is shorter than "
                "the resident initialization"
            )
        if self.tail_slot_start != (
            self.resident_tokens + self.free_slot_tokens
        ):
            raise ValueError("DSA KVIO P/D tail slot start is inconsistent")


@dataclass(frozen=True)
class DSAKVIOPDRequest:
    """D-scheduler allocation result passed to the D worker."""

    request_id: str
    manifest: DSAKVIOPDManifest
    indexer_block_ids: list[int]
    resident_block_ids: list[int]
    layer_topk_by_rank: dict[int, dict[int, list[int]]]
    initial_transport: str = DSA_PD_INITIAL_TRANSPORT_KVIO
    remote_indexer_block_ids: list[int] | None = None
    remote_resident_block_ids: list[int] | None = None
    remote_engine_id: str | None = None
    remote_request_id: str | None = None
    remote_host: str | None = None
    remote_port: int | None = None
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]] | None = None


def build_pd_resident_token_ids(
    *,
    topk_token_ids: list[int],
    stored_token_count: int,
    block_size: int,
    resident_token_count: int,
) -> list[int]:
    """Build one layer's deterministic initial resident token set.

    The last partial block remains in the independent dense-tail area and is
    therefore never inserted into lookup-managed resident slots.  Valid TopK
    positions are selected first in score order; ascending historical
    positions fill any remaining slots.  The resulting slot order is kept
    intact because KVIO supports the required discrete reads.
    """
    stored_token_count = int(stored_token_count)
    block_size = int(block_size)
    resident_token_count = int(resident_token_count)
    if stored_token_count <= 0 or block_size <= 0:
        raise ValueError("DSA resident initialization needs valid counts")
    if resident_token_count <= 0:
        raise ValueError("DSA resident initialization cannot be empty")

    dense_tail_start = (stored_token_count // block_size) * block_size
    if dense_tail_start < resident_token_count:
        raise ValueError(
            "DSA history before the dense tail is too short for "
            "resident initialization: "
            f"history={dense_tail_start}, resident={resident_token_count}"
        )

    selected: list[int] = []
    selected_set: set[int] = set()
    for raw_token_id in topk_token_ids:
        token_id = int(raw_token_id)
        if (
            token_id < 0
            or token_id >= dense_tail_start
            or token_id in selected_set
        ):
            continue
        selected.append(token_id)
        selected_set.add(token_id)
        if len(selected) == resident_token_count:
            break

    if len(selected) < resident_token_count:
        for token_id in range(dense_tail_start):
            if token_id in selected_set:
                continue
            selected.append(token_id)
            selected_set.add(token_id)
            if len(selected) == resident_token_count:
                break

    if len(selected) != resident_token_count:
        raise RuntimeError(
            "DSA could not build a complete resident initialization"
        )
    return selected


def serialize_dsa_kvio_layer_topk(
    layer_topk_by_rank: dict[int, dict[int, list[int]]],
) -> dict[str, dict[str, list[int]]]:
    return {
        str(int(rank)): {
            str(int(layer_id)): [int(token_id) for token_id in token_ids]
            for layer_id, token_ids in sorted(layers.items())
        }
        for rank, layers in sorted(layer_topk_by_rank.items())
    }


def get_dsa_kvio_layer_topk(
    kv_transfer_params: dict[str, Any] | None,
) -> dict[int, dict[int, list[int]]] | None:
    if not isinstance(kv_transfer_params, dict):
        return None
    raw_by_rank = kv_transfer_params.get(DSA_KVIO_PD_LAYER_TOPK_KEY)
    if raw_by_rank is None:
        return None
    if not isinstance(raw_by_rank, dict):
        raise TypeError("DSA KVIO per-rank layer TopK must be a dict")

    result: dict[int, dict[int, list[int]]] = {}
    for raw_rank, raw_layers in raw_by_rank.items():
        if not isinstance(raw_layers, dict):
            raise TypeError("DSA KVIO per-rank layer TopK entry must be a dict")
        rank = int(raw_rank)
        layers: dict[int, list[int]] = {}
        for raw_layer_id, raw_token_ids in raw_layers.items():
            if not isinstance(raw_token_ids, list):
                raise TypeError("DSA KVIO layer TopK positions must be a list")
            layers[int(raw_layer_id)] = [
                int(token_id) for token_id in raw_token_ids
            ]
        result[rank] = layers
    return result


def get_dsa_kvio_pd_manifest(
    kv_transfer_params: dict[str, Any] | None,
) -> DSAKVIOPDManifest | None:
    if not isinstance(kv_transfer_params, dict):
        return None
    raw_manifest = kv_transfer_params.get(DSA_KVIO_PD_MANIFEST_KEY)
    if raw_manifest is None:
        return None
    if isinstance(raw_manifest, DSAKVIOPDManifest):
        raw_manifest.validate()
        return raw_manifest
    return DSAKVIOPDManifest.from_dict(raw_manifest)
