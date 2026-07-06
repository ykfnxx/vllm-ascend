import hashlib
import json
import re
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import torch


KV_MLA_TOKEN = 0
INDEX_SIZE = 128 * 1024
SLOT_COUNT = 10 * 1024
RESIDENT_SLOT_COUNT = 8 * 1024
FREE_SLOT_COUNT = 2 * 1024
QUERY_COUNT = 2 * 1024
NOT_FOUND = -1
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
class HBMIndexLayerState:
    index: torch.Tensor
    slot_to_index: torch.Tensor
    free_slots: torch.Tensor
    free_head: torch.Tensor
    query_index: torch.Tensor
    last_query_slots: torch.Tensor
    k_nope_cache: torch.Tensor
    k_pe_cache: torch.Tensor
    resident_initialized: bool = False


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
        cache_type: int = KV_MLA_TOKEN,
        strict: bool = True,
        atol: float = 0.0,
        rtol: float = 0.0,
        lookup_op: Any | None = None,
        maintain_op: Any | None = None,
        maintain_seed: int = 0,
        index_size: int = INDEX_SIZE,
        slot_count: int = SLOT_COUNT,
        resident_slot_count: int = RESIDENT_SLOT_COUNT,
        free_slot_count: int = FREE_SLOT_COUNT,
        query_count: int = QUERY_COUNT,
        capacity: int | None = None,
        slot_table_size: int | None = None,
    ) -> None:
        self.client = client
        self.cache_type = cache_type
        self.strict = strict
        self.atol = atol
        self.rtol = rtol
        self.index_size = index_size if slot_table_size is None else slot_table_size
        self.slot_count = slot_count if capacity is None else capacity
        self.resident_slot_count = resident_slot_count
        self.free_slot_count = free_slot_count
        self.query_count = query_count
        self.capacity = self.slot_count
        self.slot_table_size = self.index_size
        self.lookup_op = lookup_op
        self.maintain_op = maintain_op
        self.maintain_seed = maintain_seed
        self._caches: dict[tuple[str, int], HBMIndexLayerState] = {}
        self._disabled_caches: set[tuple[str, int]] = set()

    def make_key(self, req_id: str, layer_id: int, token_pos: int) -> bytes:
        return make_microkv_mla_token_key(req_id, layer_id, token_pos)

    def get_slot_id(self, req_id: str, layer_id: int, token_pos: int) -> int:
        cache = self._caches.get((req_id, layer_id))
        if cache is None or token_pos >= self.index_size:
            return -1
        return int(cache.index[0, token_pos].item())

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
        resident_updates: list[tuple[str, int, int, int]] = []

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
            resident_updates.append((req_id, token_pos, prefill_len, original_slot))

        if keys:
            self.client.batch_put(self.cache_type, keys, values)
            stats.written_items = len(keys)
            for req_id, token_pos, prefill_len, original_slot in resident_updates:
                self._write_resident_prefill_token(
                    req_id=req_id,
                    layer_id=layer_id,
                    token_pos=token_pos,
                    prefill_len=prefill_len,
                    original_slot=original_slot,
                    k_nope_flat=k_nope_flat,
                    k_pe_flat=k_pe_flat,
                    k_nope_cache_template=kv_cache[0],
                    k_pe_cache_template=kv_cache[1],
                )
        return stats

    def mock_lookup_and_validate(
        self,
        layer_name: str,
        kv_cache: tuple[torch.Tensor, ...],
        topk_indices: torch.Tensor,
        attn_metadata: Any,
    ) -> LookupValidationStats:
        return self.validate_topk_with_real_hbm_index_ops(layer_name, kv_cache, topk_indices, attn_metadata)

    def validate_topk_with_real_hbm_index_ops(
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

        decode_indices_by_req: dict[int, list[int]] = {}
        for decode_token_index in range(num_decode_tokens):
            req_index = int(attn_metadata.token_req_indices_cpu[decode_token_index].item())
            decode_indices_by_req.setdefault(req_index, []).append(decode_token_index)

        for req_index, decode_token_indices in decode_indices_by_req.items():
            req_id = attn_metadata.req_ids[req_index]
            cache_key = (req_id, layer_id)
            if cache_key in self._disabled_caches:
                continue
            prefill_len = int(attn_metadata.prefill_lens_cpu[req_index].item())
            valid_query_tokens = self._collect_valid_query_tokens(topk_indices_cpu, decode_token_indices, prefill_len)
            if not valid_query_tokens:
                continue
            if len(valid_query_tokens) > self.query_count:
                raise ValueError(
                    "KV offload v0.1 query count exceeds real op QUERY_COUNT: "
                    f"req_id={req_id}, layer_id={layer_id}, count={len(valid_query_tokens)}, "
                    f"limit={self.query_count}"
                )

            state = self.get_or_create_bypass_cache(req_id, layer_id, kv_cache[0], kv_cache[1])
            query_index = self._prepare_query_index(state, valid_query_tokens)
            slot_out = self._call_lookup(state, query_index)
            token_pos_to_slot = self._load_query_tokens_to_bypass_cache(
                req_id=req_id,
                layer_id=layer_id,
                state=state,
                valid_query_tokens=valid_query_tokens,
                slot_out=slot_out,
                k_nope_cache_template=kv_cache[0],
                k_pe_cache_template=kv_cache[1],
                stats=stats,
            )

            for decode_token_index in decode_token_indices:
                for head_index in range(topk_indices_cpu.shape[1]):
                    for topk_rank in range(topk_indices_cpu.shape[2]):
                        token_pos = int(topk_indices_cpu[decode_token_index, head_index, topk_rank].item())
                        if token_pos < 0 or token_pos >= prefill_len or token_pos >= self.index_size:
                            stats.skipped_items += 1
                            continue
                        offload_slot_id = token_pos_to_slot.get(token_pos)
                        if offload_slot_id is None:
                            stats.skipped_items += 1
                            continue

                        block_id = int(block_table_cpu[req_index, token_pos // block_size].item())
                        original_slot = block_id * block_size + token_pos % block_size
                        original_k_nope = k_nope_flat[original_slot]
                        original_k_pe = k_pe_flat[original_slot]
                        bypass_k_nope = state.k_nope_cache[offload_slot_id]
                        bypass_k_pe = state.k_pe_cache[offload_slot_id]

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
                                    "KV offload v0.1 validation mismatch: "
                                    f"req_id={req_id}, layer_id={layer_id}, decode_token_index={decode_token_index}, "
                                    f"head_index={head_index}, topk_rank={topk_rank}, token_pos={token_pos}, "
                                    f"original_slot={original_slot}, offload_slot_id={offload_slot_id}, "
                                    f"max_abs_error={item_abs_error}"
                                )

            state.last_query_slots.copy_(slot_out)
            free_head = int(state.free_head[0].item())
            if free_head > 0:
                stats.evicted_items += free_head
                self._call_maintain(state)

        return stats

    def _collect_valid_query_tokens(
        self,
        topk_indices_cpu: torch.Tensor,
        decode_token_indices: list[int],
        prefill_len: int,
    ) -> list[int]:
        valid_query_tokens: list[int] = []
        seen_tokens: set[int] = set()
        for decode_token_index in decode_token_indices:
            flattened_topk = topk_indices_cpu[decode_token_index].reshape(-1)
            for token_pos_tensor in flattened_topk:
                token_pos = int(token_pos_tensor.item())
                if token_pos < 0 or token_pos >= prefill_len or token_pos >= self.index_size:
                    continue
                if token_pos not in seen_tokens:
                    seen_tokens.add(token_pos)
                    valid_query_tokens.append(token_pos)
        return valid_query_tokens

    def _prepare_query_index(self, state: HBMIndexLayerState, valid_query_tokens: list[int]) -> torch.Tensor:
        pad_token = valid_query_tokens[0]
        state.query_index.fill_(pad_token)
        state.query_index[0, : len(valid_query_tokens)] = torch.tensor(
            valid_query_tokens,
            dtype=torch.int32,
            device=state.query_index.device,
        )
        return state.query_index

    def _call_lookup(self, state: HBMIndexLayerState, query_index: torch.Tensor) -> torch.Tensor:
        lookup_op = self.lookup_op
        if lookup_op is None:
            lookup_op = torch.ops._C_ascend.asu_hbm_index_lookup
        return lookup_op(state.index, state.slot_to_index, state.free_slots, state.free_head, query_index, 1)

    def _call_maintain(self, state: HBMIndexLayerState) -> None:
        maintain_op = self.maintain_op
        if maintain_op is None:
            maintain_op = torch.ops._C_ascend.asu_hbm_index_maintain_aicpu
        maintain_op(
            state.index,
            state.slot_to_index,
            state.free_slots,
            state.free_head,
            state.last_query_slots,
            1,
            self.maintain_seed,
        )

    def _load_query_tokens_to_bypass_cache(
        self,
        req_id: str,
        layer_id: int,
        state: HBMIndexLayerState,
        valid_query_tokens: list[int],
        slot_out: torch.Tensor,
        k_nope_cache_template: torch.Tensor,
        k_pe_cache_template: torch.Tensor,
        stats: LookupValidationStats,
    ) -> dict[int, int]:
        keys = [self.make_key(req_id, layer_id, token_pos) for token_pos in valid_query_tokens]
        records = self.client.batch_get(self.cache_type, keys)
        token_pos_to_slot: dict[int, int] = {}
        for query_offset, token_pos in enumerate(valid_query_tokens):
            record = records[query_offset]
            if record is None:
                stats.missing_items += 1
                self._disabled_caches.add((req_id, layer_id))
                if self.strict:
                    raise MicroKVRecordError(
                        "KV offload v0.1 MicroKV miss after real lookup: "
                        f"req_id={req_id}, layer_id={layer_id}, token_pos={token_pos}"
                    )
                continue

            loaded_k_nope, loaded_k_pe = unpack_mla_token_record(
                record,
                expected_k_nope_shape=tuple(k_nope_cache_template.shape[2:]),
                expected_k_pe_shape=tuple(k_pe_cache_template.shape[2:]),
                expected_dtype=k_nope_cache_template.dtype,
                device=k_nope_cache_template.device,
            )
            slot_id = int(slot_out[0, query_offset].item())
            state.k_nope_cache[slot_id].copy_(loaded_k_nope)
            state.k_pe_cache[slot_id].copy_(loaded_k_pe)
            token_pos_to_slot[token_pos] = slot_id
            stats.loaded_items += 1
        return token_pos_to_slot

    def _write_resident_prefill_token(
        self,
        req_id: str,
        layer_id: int,
        token_pos: int,
        prefill_len: int,
        original_slot: int,
        k_nope_flat: torch.Tensor,
        k_pe_flat: torch.Tensor,
        k_nope_cache_template: torch.Tensor,
        k_pe_cache_template: torch.Tensor,
    ) -> None:
        resident_count = min(prefill_len, self.resident_slot_count, self.index_size)
        if token_pos >= resident_count:
            return

        state = self.get_or_create_bypass_cache(req_id, layer_id, k_nope_cache_template, k_pe_cache_template)
        slot_id = token_pos
        state.index[0, token_pos] = slot_id
        state.slot_to_index[0, slot_id] = token_pos
        state.k_nope_cache[slot_id].copy_(k_nope_flat[original_slot])
        state.k_pe_cache[slot_id].copy_(k_pe_flat[original_slot])
        state.resident_initialized = True

    def get_or_create_bypass_cache(
        self,
        req_id: str,
        layer_id: int,
        k_nope_cache_template: torch.Tensor,
        k_pe_cache_template: torch.Tensor,
    ) -> HBMIndexLayerState:
        cache_key = (req_id, layer_id)
        cache = self._caches.get(cache_key)
        if cache is not None:
            return cache

        device = k_nope_cache_template.device
        free_slots = torch.arange(
            self.resident_slot_count,
            self.resident_slot_count + self.free_slot_count,
            dtype=torch.int32,
            device=device,
        ).view(1, self.free_slot_count)
        cache = HBMIndexLayerState(
            index=torch.full((1, self.index_size), NOT_FOUND, dtype=torch.int32, device=device),
            slot_to_index=torch.full((1, self.slot_count), NOT_FOUND, dtype=torch.int32, device=device),
            free_slots=free_slots,
            free_head=torch.zeros((1,), dtype=torch.int32, device=device),
            query_index=torch.empty((1, self.query_count), dtype=torch.int32, device=device),
            last_query_slots=torch.empty((1, self.query_count), dtype=torch.int32, device=device),
            k_nope_cache=torch.empty(
                (self.slot_count, *k_nope_cache_template.shape[2:]),
                dtype=k_nope_cache_template.dtype,
                device=device,
            ),
            k_pe_cache=torch.empty(
                (self.slot_count, *k_pe_cache_template.shape[2:]),
                dtype=k_pe_cache_template.dtype,
                device=device,
            ),
        )
        self._caches[cache_key] = cache
        return cache
