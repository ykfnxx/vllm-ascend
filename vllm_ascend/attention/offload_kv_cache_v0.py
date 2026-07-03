import hashlib
import json
import re
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import torch


KV_MLA_TOKEN = 0
_RECORD_MAGIC = b"MKV0"
_RECORD_VERSION = 1
_RECORD_HEADER_LENGTH = struct.Struct("<I")
_LAYER_ID_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_TORCH_DTYPE_TO_RECORD_DTYPE = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_RECORD_DTYPE_TO_TORCH_DTYPE = {name: dtype for dtype, name in _TORCH_DTYPE_TO_RECORD_DTYPE.items()}


class MicroKVRecordError(ValueError):
    """Serialized MLA token record is not compatible with the current layer."""


class OffloadKVCacheV0MismatchError(AssertionError):
    """Bypass cache data differs from the original vLLM KV cache data."""


@dataclass
class PrefillPersistStats:
    written_items: int = 0
    skipped_items: int = 0


@dataclass
class LookupValidationStats:
    checked_items: int = 0
    skipped_items: int = 0
    loaded_items: int = 0
    evicted_items: int = 0
    missing_items: int = 0
    mismatch_items: int = 0
    max_abs_error: float = 0.0


@dataclass
class BypassLayerCache:
    slot_table: torch.Tensor
    k_nope_cache: torch.Tensor
    k_pe_cache: torch.Tensor
    token_pos_by_slot: list[int | None]
    next_unallocated_slot: int = 0
    next_eviction_slot: int = 0

    def allocate_slot_for_token(self, token_pos: int) -> tuple[int, int | None]:
        if self.next_unallocated_slot < len(self.token_pos_by_slot):
            slot_id = self.next_unallocated_slot
            self.next_unallocated_slot += 1
            self.token_pos_by_slot[slot_id] = token_pos
            return slot_id, None

        slot_id = self.next_eviction_slot
        self.next_eviction_slot = (self.next_eviction_slot + 1) % len(self.token_pos_by_slot)
        evicted_token_pos = self.token_pos_by_slot[slot_id]
        self.token_pos_by_slot[slot_id] = token_pos
        return slot_id, evicted_token_pos


def parse_layer_id(layer_name: str) -> int:
    match = _LAYER_ID_PATTERN.search(layer_name)
    if match is None:
        raise ValueError(f"layer_name does not contain a model layer id: {layer_name}")
    return int(match.group(1))


def make_microkv_mla_token_key(req_id: str, layer_id: int, token_pos: int) -> bytes:
    req_bytes = hashlib.sha256(req_id.encode()).digest()[:16]
    return (
        req_bytes
        + struct.pack("<I", int(layer_id))
        + struct.pack("<I", int(token_pos))
        + struct.pack("<B", KV_MLA_TOKEN)
        + b"\x00" * 7
    )


def pack_mla_token_record(k_nope: torch.Tensor, k_pe: torch.Tensor) -> bytes:
    if k_nope.dtype != k_pe.dtype:
        raise MicroKVRecordError(f"k_nope dtype {k_nope.dtype} differs from k_pe dtype {k_pe.dtype}")
    if k_nope.dtype not in _TORCH_DTYPE_TO_RECORD_DTYPE:
        raise MicroKVRecordError(f"unsupported MLA token dtype: {k_nope.dtype}")

    k_nope_cpu = k_nope.detach().cpu().contiguous()
    k_pe_cpu = k_pe.detach().cpu().contiguous()
    k_nope_payload = k_nope_cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    k_pe_payload = k_pe_cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    payload = k_nope_payload + k_pe_payload
    header = {
        "magic": _RECORD_MAGIC.decode("ascii"),
        "version": _RECORD_VERSION,
        "dtype": _TORCH_DTYPE_TO_RECORD_DTYPE[k_nope.dtype],
        "k_nope_shape": list(k_nope_cpu.shape),
        "k_pe_shape": list(k_pe_cpu.shape),
        "k_nope_nbytes": len(k_nope_payload),
        "k_pe_nbytes": len(k_pe_payload),
        "payload_checksum": zlib.crc32(payload) & 0xFFFFFFFF,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    return _RECORD_MAGIC + _RECORD_HEADER_LENGTH.pack(len(header_bytes)) + header_bytes + payload


def unpack_mla_token_record(
    record: bytes,
    expected_k_nope_shape: tuple[int, ...] | None = None,
    expected_k_pe_shape: tuple[int, ...] | None = None,
    expected_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if record[: len(_RECORD_MAGIC)] != _RECORD_MAGIC:
        raise MicroKVRecordError("MLA token record has invalid magic")

    header_start = len(_RECORD_MAGIC) + _RECORD_HEADER_LENGTH.size
    header_length = _RECORD_HEADER_LENGTH.unpack(record[len(_RECORD_MAGIC) : header_start])[0]
    header_end = header_start + header_length
    header = json.loads(record[header_start:header_end].decode("ascii"))
    payload = record[header_end:]

    if header["magic"] != _RECORD_MAGIC.decode("ascii"):
        raise MicroKVRecordError("MLA token record header has invalid magic")
    if header["version"] != _RECORD_VERSION:
        raise MicroKVRecordError(f"unsupported MLA token record version: {header['version']}")

    dtype = _RECORD_DTYPE_TO_TORCH_DTYPE[header["dtype"]]
    k_nope_shape = tuple(header["k_nope_shape"])
    k_pe_shape = tuple(header["k_pe_shape"])
    k_nope_nbytes = int(header["k_nope_nbytes"])
    k_pe_nbytes = int(header["k_pe_nbytes"])

    if expected_dtype is not None and dtype != expected_dtype:
        raise MicroKVRecordError(f"record dtype {dtype} does not match expected dtype {expected_dtype}")
    if expected_k_nope_shape is not None and k_nope_shape != expected_k_nope_shape:
        raise MicroKVRecordError(
            f"record k_nope shape {k_nope_shape} does not match expected shape {expected_k_nope_shape}"
        )
    if expected_k_pe_shape is not None and k_pe_shape != expected_k_pe_shape:
        raise MicroKVRecordError(f"record k_pe shape {k_pe_shape} does not match expected shape {expected_k_pe_shape}")
    if len(payload) != k_nope_nbytes + k_pe_nbytes:
        raise MicroKVRecordError("MLA token record payload length does not match header")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != int(header["payload_checksum"]):
        raise MicroKVRecordError("MLA token record payload checksum mismatch")

    k_nope_bytes = payload[:k_nope_nbytes]
    k_pe_bytes = payload[k_nope_nbytes:]
    k_nope = torch.frombuffer(bytearray(k_nope_bytes), dtype=torch.uint8).view(dtype).reshape(k_nope_shape)
    k_pe = torch.frombuffer(bytearray(k_pe_bytes), dtype=torch.uint8).view(dtype).reshape(k_pe_shape)
    if device is not None:
        k_nope = k_nope.to(device)
        k_pe = k_pe.to(device)
    return k_nope, k_pe


class OffloadKVCacheV0Manager:
    def __init__(
        self,
        client: Any,
        capacity: int,
        slot_table_size: int = 128 * 1024,
        cache_type: int = KV_MLA_TOKEN,
        strict: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> None:
        self.client = client
        self.capacity = capacity
        self.slot_table_size = slot_table_size
        self.cache_type = cache_type
        self.strict = strict
        self.atol = atol
        self.rtol = rtol
        self._caches: dict[tuple[str, int], BypassLayerCache] = {}

    def make_key(self, req_id: str, layer_id: int, token_pos: int) -> bytes:
        return make_microkv_mla_token_key(req_id, layer_id, token_pos)

    def get_slot_id(self, req_id: str, layer_id: int, token_pos: int) -> int:
        cache = self._caches.get((req_id, layer_id))
        if cache is None or token_pos >= self.slot_table_size:
            return -1
        return int(cache.slot_table[token_pos].item())

    def persist_prefill_kv_to_microkv(
        self,
        layer_name: str,
        kv_cache: tuple[torch.Tensor, ...],
        slot_mapping: torch.Tensor,
        attn_metadata: Any,
    ) -> PrefillPersistStats:
        layer_id = parse_layer_id(layer_name)
        k_nope_flat = kv_cache[0].view(-1, *kv_cache[0].shape[2:])
        k_pe_flat = kv_cache[1].view(-1, *kv_cache[1].shape[2:])
        slot_mapping_cpu = slot_mapping.detach().cpu()
        stats = PrefillPersistStats()
        keys: list[bytes] = []
        values: list[bytes] = []

        state_name = getattr(attn_metadata.attn_state, "name", str(attn_metadata.attn_state))
        if state_name in {"DecodeOnly", "SpecDecoding"}:
            stats.skipped_items = int(getattr(attn_metadata, "num_actual_tokens", 0))
            return stats

        num_tokens = int(attn_metadata.num_actual_tokens)
        for token_index in range(num_tokens):
            req_index = int(attn_metadata.token_req_indices_cpu[token_index].item())
            token_pos = int(attn_metadata.token_positions_cpu[token_index].item())
            prefill_len = int(attn_metadata.prefill_lens_cpu[req_index].item())
            original_slot = int(slot_mapping_cpu[token_index].item())
            if token_pos >= prefill_len or original_slot < 0:
                stats.skipped_items += 1
                continue

            req_id = attn_metadata.req_ids[req_index]
            keys.append(self.make_key(req_id, layer_id, token_pos))
            values.append(pack_mla_token_record(k_nope_flat[original_slot], k_pe_flat[original_slot]))

        if keys:
            self.client.batch_put(self.cache_type, keys, values)
            stats.written_items = len(keys)
        return stats

    def mock_lookup_and_validate(
        self,
        layer_name: str,
        kv_cache: tuple[torch.Tensor, ...],
        topk_indices: torch.Tensor,
        attn_metadata: Any,
    ) -> LookupValidationStats:
        layer_id = parse_layer_id(layer_name)
        stats = LookupValidationStats()
        k_nope_flat = kv_cache[0].view(-1, *kv_cache[0].shape[2:])
        k_pe_flat = kv_cache[1].view(-1, *kv_cache[1].shape[2:])
        topk_indices_cpu = topk_indices.detach().cpu()
        block_table_cpu = attn_metadata.block_table.detach().cpu()
        block_size = int(kv_cache[0].shape[1])
        num_decode_tokens = int(attn_metadata.num_decode_tokens)
        if num_decode_tokens == 0:
            return stats

        for decode_token_index in range(num_decode_tokens):
            req_index = int(attn_metadata.token_req_indices_cpu[decode_token_index].item())
            req_id = attn_metadata.req_ids[req_index]
            prefill_len = int(attn_metadata.prefill_lens_cpu[req_index].item())
            for head_index in range(topk_indices_cpu.shape[1]):
                for topk_rank in range(topk_indices_cpu.shape[2]):
                    token_pos = int(topk_indices_cpu[decode_token_index, head_index, topk_rank].item())
                    if token_pos < 0 or token_pos >= prefill_len:
                        stats.skipped_items += 1
                        continue

                    bypass_cache, offload_slot_id = self.load_missing_token_into_bypass_cache(
                        req_id=req_id,
                        layer_id=layer_id,
                        token_pos=token_pos,
                        k_nope_cache_template=kv_cache[0],
                        k_pe_cache_template=kv_cache[1],
                        stats=stats,
                    )
                    if offload_slot_id < 0:
                        stats.skipped_items += 1
                        stats.missing_items += 1
                        continue

                    block_id = int(block_table_cpu[req_index, token_pos // block_size].item())
                    original_slot = block_id * block_size + token_pos % block_size
                    original_k_nope = k_nope_flat[original_slot]
                    original_k_pe = k_pe_flat[original_slot]
                    bypass_k_nope = bypass_cache.k_nope_cache[offload_slot_id]
                    bypass_k_pe = bypass_cache.k_pe_cache[offload_slot_id]

                    nope_abs_error = torch.max(torch.abs(original_k_nope - bypass_k_nope)).item()
                    pe_abs_error = torch.max(torch.abs(original_k_pe - bypass_k_pe)).item()
                    item_abs_error = max(float(nope_abs_error), float(pe_abs_error))
                    stats.max_abs_error = max(stats.max_abs_error, item_abs_error)
                    stats.checked_items += 1
                    k_nope_matches = torch.allclose(original_k_nope, bypass_k_nope, atol=self.atol, rtol=self.rtol)
                    k_pe_matches = torch.allclose(original_k_pe, bypass_k_pe, atol=self.atol, rtol=self.rtol)
                    if not k_nope_matches or not k_pe_matches:
                        stats.mismatch_items += 1
                        if self.strict:
                            raise OffloadKVCacheV0MismatchError(
                                "KV offload v0 validation mismatch: "
                                f"req_id={req_id}, layer_id={layer_id}, decode_token_index={decode_token_index}, "
                                f"head_index={head_index}, topk_rank={topk_rank}, token_pos={token_pos}, "
                                f"original_slot={original_slot}, offload_slot_id={offload_slot_id}, "
                                f"max_abs_error={item_abs_error}"
                            )

        return stats

    def load_missing_token_into_bypass_cache(
        self,
        req_id: str,
        layer_id: int,
        token_pos: int,
        k_nope_cache_template: torch.Tensor,
        k_pe_cache_template: torch.Tensor,
        stats: LookupValidationStats,
    ) -> tuple[BypassLayerCache, int]:
        bypass_cache = self.get_or_create_bypass_cache(req_id, layer_id, k_nope_cache_template, k_pe_cache_template)
        existing_slot_id = int(bypass_cache.slot_table[token_pos].item())
        if existing_slot_id >= 0:
            return bypass_cache, existing_slot_id

        record = self.client.batch_get(self.cache_type, [self.make_key(req_id, layer_id, token_pos)])[0]
        if record is None:
            return bypass_cache, -1

        loaded_k_nope, loaded_k_pe = unpack_mla_token_record(
            record,
            expected_k_nope_shape=tuple(k_nope_cache_template.shape[2:]),
            expected_k_pe_shape=tuple(k_pe_cache_template.shape[2:]),
            expected_dtype=k_nope_cache_template.dtype,
            device=k_nope_cache_template.device,
        )
        slot_id, evicted_token_pos = bypass_cache.allocate_slot_for_token(token_pos)
        if evicted_token_pos is not None:
            bypass_cache.slot_table[evicted_token_pos] = -1
            stats.evicted_items += 1

        bypass_cache.slot_table[token_pos] = slot_id
        bypass_cache.k_nope_cache[slot_id].copy_(loaded_k_nope)
        bypass_cache.k_pe_cache[slot_id].copy_(loaded_k_pe)
        stats.loaded_items += 1
        return bypass_cache, slot_id

    def get_or_create_bypass_cache(
        self,
        req_id: str,
        layer_id: int,
        k_nope_cache_template: torch.Tensor,
        k_pe_cache_template: torch.Tensor,
    ) -> BypassLayerCache:
        cache_key = (req_id, layer_id)
        cache = self._caches.get(cache_key)
        if cache is not None:
            return cache

        cache = BypassLayerCache(
            slot_table=torch.full((self.slot_table_size,), -1, dtype=torch.int32, device="cpu"),
            k_nope_cache=torch.empty(
                (self.capacity, *k_nope_cache_template.shape[2:]),
                dtype=k_nope_cache_template.dtype,
                device=k_nope_cache_template.device,
            ),
            k_pe_cache=torch.empty(
                (self.capacity, *k_pe_cache_template.shape[2:]),
                dtype=k_pe_cache_template.dtype,
                device=k_pe_cache_template.device,
            ),
            token_pos_by_slot=[None] * self.capacity,
        )
        self._caches[cache_key] = cache
        return cache
