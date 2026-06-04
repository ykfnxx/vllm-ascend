#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import os
from dataclasses import dataclass

import torch

INVALID = -1
ASU_ONLY = 0
HBM_RESIDENT = 1
TAIL_HBM = 2

ASU_KV_CACHE_ENABLE_ENV = "VLLM_ASCEND_ENABLE_ASU_KV_CACHE"
ASU_KV_CACHE_SLOT_COUNT_ENV = "VLLM_ASCEND_ASU_KV_CACHE_SLOTS"


def is_asu_kv_cache_enabled() -> bool:
    value = os.getenv(ASU_KV_CACHE_ENABLE_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def get_configured_managed_slot_count() -> int | None:
    value = os.getenv(ASU_KV_CACHE_SLOT_COUNT_ENV)
    if value is None or value == "":
        return None
    slot_count = int(value)
    if slot_count <= 0:
        raise ValueError(f"{ASU_KV_CACHE_SLOT_COUNT_ENV} must be positive")
    return slot_count


def _flatten_pa_cache(cache: torch.Tensor) -> torch.Tensor:
    if cache.dim() < 2:
        raise ValueError("PA cache must have block and block-size dimensions")
    return cache.reshape(cache.shape[0] * cache.shape[1], *cache.shape[2:])


@dataclass
class ASULayerKVCacheContext:
    layer_name: str
    managed_kv_cache_0: torch.Tensor
    managed_kv_cache_1: torch.Tensor
    token_state: torch.Tensor
    asu_record_addr: torch.Tensor
    hbm_slot_of_token: torch.Tensor
    slot_owner_token: torch.Tensor
    free_slot_stack: torch.Tensor
    free_slot_count: torch.Tensor
    max_seq_len: int
    managed_slot_count: int
    block_size: int
    req_id: str | None = None
    actual_seq_len: int = 0
    managed_prefix_len: int = 0

    def original_kv_cache_views(
        self,
        kv_cache: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _flatten_pa_cache(kv_cache[0]), _flatten_pa_cache(kv_cache[1])

    def reset_state(
        self,
        req_id: str,
        actual_seq_len: int,
        managed_prefix_len: int,
        original_block_table: torch.Tensor,
    ) -> None:
        if actual_seq_len < 0:
            raise ValueError("actual_seq_len must be non-negative")
        if managed_prefix_len < 0:
            raise ValueError("managed_prefix_len must be non-negative")
        if managed_prefix_len > actual_seq_len:
            raise ValueError("managed_prefix_len must not exceed actual_seq_len")
        if actual_seq_len > self.max_seq_len:
            raise RuntimeError("ASU max_seq_len does not cover actual_seq_len")

        self.req_id = req_id
        self.actual_seq_len = actual_seq_len
        self.managed_prefix_len = managed_prefix_len

        self.token_state.fill_(INVALID)
        self.asu_record_addr.fill_(INVALID)
        self.hbm_slot_of_token.fill_(INVALID)
        self.slot_owner_token.fill_(INVALID)
        self.free_slot_stack.copy_(
            torch.arange(
                self.managed_slot_count,
                device=self.free_slot_stack.device,
                dtype=self.free_slot_stack.dtype,
            )
        )
        self.free_slot_count.reshape(-1)[0] = self.managed_slot_count

        if managed_prefix_len > 0:
            positions = torch.arange(
                managed_prefix_len,
                device=self.asu_record_addr.device,
                dtype=torch.int64,
            )
            logical_blocks = torch.div(
                positions,
                self.block_size,
                rounding_mode="floor",
            )
            offsets = positions % self.block_size
            block_table = original_block_table.reshape(-1).to(
                device=self.asu_record_addr.device,
                dtype=torch.int64,
            )
            if logical_blocks[-1].item() >= block_table.numel():
                raise RuntimeError("original_block_table does not cover ASU prefix")
            source_slots = block_table[logical_blocks] * self.block_size + offsets
            self.token_state[:managed_prefix_len].fill_(ASU_ONLY)
            self.asu_record_addr[:managed_prefix_len].copy_(
                source_slots.to(self.asu_record_addr.dtype)
            )

        if managed_prefix_len < actual_seq_len:
            self.token_state[managed_prefix_len:actual_seq_len].fill_(TAIL_HBM)

    def mark_decode_token(self, token_id: int) -> None:
        if token_id < 0 or token_id >= self.max_seq_len:
            raise RuntimeError("decode token_id is outside ASU max_seq_len")
        self.actual_seq_len = max(self.actual_seq_len, token_id + 1)
        self.token_state[token_id] = TAIL_HBM


class ASUFullKVCacheManagerFunctional:
    def __init__(
        self,
        enabled: bool,
        max_seq_len: int,
        block_size: int,
        contexts: dict[str, ASULayerKVCacheContext],
    ) -> None:
        self.enabled = enabled
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self._contexts = contexts

    @classmethod
    def from_kv_caches(
        cls,
        kv_caches: dict[str, tuple[torch.Tensor, ...]],
        max_seq_len: int,
        block_size: int,
        managed_slot_count: int | None = None,
        enabled: bool | None = None,
    ) -> "ASUFullKVCacheManagerFunctional":
        is_enabled = is_asu_kv_cache_enabled() if enabled is None else enabled
        if not is_enabled:
            return cls(False, max_seq_len, block_size, {})

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        slot_count = (
            managed_slot_count or get_configured_managed_slot_count()
            or max_seq_len
        )
        if slot_count <= 0:
            raise ValueError("managed_slot_count must be positive")

        contexts: dict[str, ASULayerKVCacheContext] = {}
        for layer_name, kv_cache in kv_caches.items():
            if not isinstance(kv_cache, tuple) or len(kv_cache) < 2:
                continue
            original_kv_cache_0 = kv_cache[0]
            original_kv_cache_1 = kv_cache[1]
            if original_kv_cache_0.dim() < 3 or original_kv_cache_1.dim() < 3:
                continue

            device = original_kv_cache_0.device
            managed_kv_cache_0 = torch.empty(
                (slot_count, *original_kv_cache_0.shape[2:]),
                device=device,
                dtype=original_kv_cache_0.dtype,
            )
            managed_kv_cache_1 = torch.empty(
                (slot_count, *original_kv_cache_1.shape[2:]),
                device=original_kv_cache_1.device,
                dtype=original_kv_cache_1.dtype,
            )

            contexts[layer_name] = ASULayerKVCacheContext(
                layer_name=layer_name,
                managed_kv_cache_0=managed_kv_cache_0,
                managed_kv_cache_1=managed_kv_cache_1,
                token_state=torch.full(
                    (max_seq_len,),
                    INVALID,
                    device=device,
                    dtype=torch.int32,
                ),
                asu_record_addr=torch.full(
                    (max_seq_len,),
                    INVALID,
                    device=device,
                    dtype=torch.int32,
                ),
                hbm_slot_of_token=torch.full(
                    (max_seq_len,),
                    INVALID,
                    device=device,
                    dtype=torch.int32,
                ),
                slot_owner_token=torch.full(
                    (slot_count,),
                    INVALID,
                    device=device,
                    dtype=torch.int32,
                ),
                free_slot_stack=torch.arange(
                    slot_count,
                    device=device,
                    dtype=torch.int32,
                ),
                free_slot_count=torch.tensor(
                    slot_count,
                    device=device,
                    dtype=torch.int32,
                ),
                max_seq_len=max_seq_len,
                managed_slot_count=slot_count,
                block_size=block_size,
            )

        return cls(True, max_seq_len, block_size, contexts)

    def has_layer(self, layer_name: str) -> bool:
        return layer_name in self._contexts

    def get_layer_context(self, layer_name: str) -> ASULayerKVCacheContext:
        return self._contexts[layer_name]

    def reset_single_request(
        self,
        req_id: str,
        actual_seq_len: int,
        managed_prefix_len: int,
        original_block_table: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        for context in self._contexts.values():
            context.reset_state(
                req_id,
                actual_seq_len,
                managed_prefix_len,
                original_block_table,
            )

    def mark_decode_token(self, token_id: int) -> None:
        if not self.enabled:
            return
        for context in self._contexts.values():
            context.mark_decode_token(token_id)
