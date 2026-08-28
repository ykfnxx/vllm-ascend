# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from .hot_cache import HotCacheLayout

_STORAGE_KEY_DOMAIN = b"dsa-offload-mla-v1"
_INT63_MASK = (1 << 63) - 1


class IOBackend(Protocol):
    def register_put_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None: ...

    def register_get_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None: ...

    def finalize_registration(self) -> None: ...

    def put_blocks(
        self,
        *,
        layer_id: int,
        storage_ids: torch.Tensor,
        source_block_ids: torch.Tensor,
    ) -> None: ...

    def get_tokens(
        self,
        *,
        layer_id: int,
        storage_ids: torch.Tensor,
        token_offsets: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> None: ...

    def close(self) -> None: ...


def make_storage_id(block_hash: bytes, layer_id: int) -> int:
    digest = hashlib.sha256(_STORAGE_KEY_DOMAIN + block_hash + layer_id.to_bytes(4, "big", signed=False)).digest()
    storage_id = int.from_bytes(digest[:8], "big") & _INT63_MASK
    return storage_id or 1


def make_storage_ids(
    block_hashes: Sequence[bytes],
    layer_id: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.tensor(
        [make_storage_id(block_hash, layer_id) for block_hash in block_hashes],
        dtype=torch.int64,
        device=device,
    )


def require_block_hashes(
    block_hashes: Sequence[bytes],
    required_blocks: int,
    *,
    context: str,
) -> None:
    available_blocks = len(block_hashes)
    if available_blocks < required_blocks:
        raise RuntimeError(
            f"DSA Offload {context} requires {required_blocks} block hashes, "
            f"but only {available_blocks} are available."
        )


class MockIOBackend:
    def register_put_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None:
        return

    def register_get_cache(
        self,
        *,
        layer_id: int,
        block_size: int,
        cache_planes: tuple[torch.Tensor, ...],
    ) -> None:
        return

    def finalize_registration(self) -> None:
        return

    def put_blocks(
        self,
        *,
        layer_id: int,
        storage_ids: torch.Tensor,
        source_block_ids: torch.Tensor,
    ) -> None:
        return

    def get_tokens(
        self,
        *,
        layer_id: int,
        storage_ids: torch.Tensor,
        token_offsets: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> None:
        return

    def close(self) -> None:
        return


def create_io_backend(
    io_backend: str,
    kvio_model_id: int,
    layout: "HotCacheLayout | None" = None,
) -> IOBackend:
    if io_backend == "mock":
        return MockIOBackend()
    if io_backend == "kvio":
        from .kvio import KVIOBackend

        return KVIOBackend(kvio_model_id)
    if io_backend == "kvgather_sim":
        if layout is None:
            raise ValueError("kvgather_sim requires a Hot Cache layout")
        from .kvgather_sim import KVGatherSimBackend

        return KVGatherSimBackend(layout)
    raise ValueError(f"Unsupported DSA Offload IO backend: {io_backend}")
