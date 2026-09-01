# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .metadata import DecodeHashContext, make_block_key

if TYPE_CHECKING:
    import torch

    from .lookup import DSAOffloadBatch

@dataclass
class DecodeBlockHashState:
    block_size: int
    block_hasher: Callable[[bytes | None, Sequence[int], tuple[Any, ...] | None], bytes]
    contexts: dict[str, DecodeHashContext] = field(default_factory=dict)
    canonical_tails: dict[str, tuple[int, bytes]] = field(
        default_factory=dict
    )

    @classmethod
    def create(cls, block_size: int, hash_name: str) -> "DecodeBlockHashState":
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import (
            hash_block_tokens,
            init_none_hash,
        )

        hash_function = get_hash_fn_by_name(hash_name)
        init_none_hash(hash_function)

        def block_hasher(
            parent_hash: bytes | None,
            token_ids: Sequence[int],
            extra_keys: tuple[Any, ...] | None,
        ) -> bytes:
            return bytes(
                hash_block_tokens(
                    hash_function,
                    parent_hash,
                    token_ids,
                    extra_keys,
                )
            )

        return cls(block_size, block_hasher)

    def update_contexts(
        self,
        contexts: Mapping[str, DecodeHashContext],
    ) -> None:
        self.contexts.update(contexts)

    def resolve(
        self,
        *,
        batch: "DSAOffloadBatch",
        query_token_ids: Sequence[int] | "torch.Tensor",
        committed_block_keys: dict[str, list[int]],
        request_index: int,
        logical_block: int,
    ) -> int:
        request_id = batch.request_ids[request_index]
        committed = committed_block_keys[request_id]
        if logical_block < len(committed):
            return committed[logical_block]
        if logical_block != len(committed):
            raise RuntimeError(
                "DSA Offload Decode cannot resolve a non-contiguous block key "
                f"for request {request_id}: block={logical_block}, "
                f"committed={len(committed)}."
            )

        context = self.contexts.get(request_id)
        if context is None:
            raise RuntimeError(
                f"DSA Offload Decode hash context is missing for request {request_id}."
            )
        block_index, parent_hash, known_token_ids, extra_keys = context
        if block_index != logical_block:
            raise RuntimeError(
                "DSA Offload Decode hash context is stale for request "
                f"{request_id}: context_block={block_index}, "
                f"required_block={logical_block}."
            )

        previous_tail = self.canonical_tails.get(request_id)
        if (
            previous_tail is not None
            and previous_tail[0] == logical_block - 1
            and parent_hash != previous_tail[1]
        ):
            raise RuntimeError(
                "DSA Offload Decode parent block hash diverged for request "
                f"{request_id} at block {logical_block}."
            )

        tokens: list[int | None] = [None] * self.block_size
        for offset, token_id in enumerate(known_token_ids):
            if offset >= self.block_size:
                break
            tokens[offset] = token_id

        begin, end = batch.query_ranges[request_index]
        positions = batch.query_positions_cpu[begin:end]
        input_ids = query_token_ids[begin:end]
        block_start = logical_block * self.block_size
        for position, token_id in zip(positions, input_ids):
            offset = int(position) - block_start
            if not 0 <= offset < self.block_size:
                continue
            existing = tokens[offset]
            if existing is not None and existing != int(token_id):
                raise RuntimeError(
                    "DSA Offload Decode token context diverged for request "
                    f"{request_id} at position {position}."
                )
            tokens[offset] = int(token_id)

        missing = [offset for offset, token_id in enumerate(tokens) if token_id is None]
        if missing:
            raise RuntimeError(
                "DSA Offload Decode cannot build the completed block hash for "
                f"request {request_id}: block={logical_block}, "
                f"missing_token_offsets={missing[:8]}."
            )

        block_hash = self.block_hasher(
            parent_hash,
            [int(token_id) for token_id in tokens],
            extra_keys,
        )
        block_key = make_block_key(block_hash)
        committed.append(block_key)
        self.canonical_tails[request_id] = (logical_block, block_hash)
        return block_key

    def release(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self.contexts.pop(request_id, None)
            self.canonical_tails.pop(request_id, None)
