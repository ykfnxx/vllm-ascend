# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Single-host shared-memory payloads for DSA Sparse P/D handoff."""

from __future__ import annotations

import math
import mmap
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DSA_SPARSE_SHARED_MEMORY_ROOT = Path("/dev/shm")
DSA_SPARSE_SHARED_MEMORY_PREFIX = "vllm_ascend_dsa_sparse_"

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
    "uint8": torch.uint8,
}
for _optional_dtype_name in (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e8m0fnu",
):
    _optional_dtype = getattr(torch, _optional_dtype_name, None)
    if isinstance(_optional_dtype, torch.dtype):
        _DTYPE_BY_NAME[_optional_dtype_name] = _optional_dtype
_NAME_BY_DTYPE = {dtype: name for name, dtype in _DTYPE_BY_NAME.items()}

DSA_SPARSE_SHARED_MEMORY_CACHE_KINDS = frozenset({"indexer", "mtp_draft"})


@dataclass(frozen=True)
class DSASparseSharedMemoryPlane:
    """One tensor plane packed into a shared-memory payload."""

    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    block_scale: int = 1

    def __post_init__(self) -> None:
        if self.offset < 0 or self.nbytes <= 0:
            raise ValueError("DSA Sparse shared-memory plane range is invalid")
        if self.dtype not in _DTYPE_BY_NAME:
            raise ValueError(f"Unsupported DSA Sparse shared-memory dtype: {self.dtype!r}")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError("DSA Sparse shared-memory plane shape is invalid")
        if self.block_scale <= 0:
            raise ValueError("DSA Sparse shared-memory block scale must be positive")
        expected_nbytes = torch.empty((), dtype=_DTYPE_BY_NAME[self.dtype]).element_size() * math.prod(self.shape)
        if expected_nbytes != self.nbytes:
            raise ValueError(
                "DSA Sparse shared-memory plane byte length does not match "
                f"its dtype and shape: expected={expected_nbytes}, actual={self.nbytes}."
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_BY_NAME[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "nbytes": self.nbytes,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "block_scale": self.block_scale,
        }

    @classmethod
    def from_dict(cls, raw: object) -> DSASparseSharedMemoryPlane:
        if not isinstance(raw, dict):
            raise TypeError("DSA Sparse shared-memory plane must be a dictionary")
        shape = raw.get("shape")
        if not isinstance(shape, (list, tuple)) or any(
            isinstance(dim, bool) or not isinstance(dim, int) for dim in shape
        ):
            raise TypeError("DSA Sparse shared-memory plane shape is invalid")
        integer_fields = {
            "offset": raw.get("offset"),
            "nbytes": raw.get("nbytes"),
            "block_scale": raw.get("block_scale", 1),
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields.values()):
            raise TypeError("DSA Sparse shared-memory plane integers are invalid")
        dtype = raw.get("dtype")
        if not isinstance(dtype, str):
            raise TypeError("DSA Sparse shared-memory plane dtype must be a string")
        return cls(
            offset=integer_fields["offset"],
            nbytes=integer_fields["nbytes"],
            dtype=dtype,
            shape=tuple(shape),
            block_scale=integer_fields["block_scale"],
        )


@dataclass(frozen=True)
class DSASparseSharedMemoryPayload:
    """Serializable description of one request/layer shared-memory bundle."""

    name: str
    size: int
    cache_kind: str
    cache_layer_name: str
    cache_planes: tuple[DSASparseSharedMemoryPlane, ...]
    tail_planes: tuple[DSASparseSharedMemoryPlane, ...]

    def __post_init__(self) -> None:
        if not self.name.startswith(DSA_SPARSE_SHARED_MEMORY_PREFIX) or Path(self.name).name != self.name:
            raise ValueError("DSA Sparse shared-memory object name is invalid")
        if self.size <= 0:
            raise ValueError("DSA Sparse shared-memory payload size must be positive")
        if self.cache_kind not in DSA_SPARSE_SHARED_MEMORY_CACHE_KINDS:
            raise ValueError(f"DSA Sparse shared-memory cache kind is invalid: {self.cache_kind!r}")
        if not self.cache_layer_name or not self.cache_planes:
            raise ValueError("DSA Sparse shared-memory cache layer and planes are required")
        if self.tail_planes and self.cache_kind != "indexer":
            raise ValueError("Only an Indexer payload may bundle a Main Tail")
        for plane in (*self.cache_planes, *self.tail_planes):
            if plane.offset + plane.nbytes > self.size:
                raise ValueError("DSA Sparse shared-memory plane exceeds its payload size")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "cache_kind": self.cache_kind,
            "cache_layer_name": self.cache_layer_name,
            "cache_planes": [plane.to_dict() for plane in self.cache_planes],
            "tail_planes": [plane.to_dict() for plane in self.tail_planes],
        }

    @classmethod
    def from_dict(cls, raw: object) -> DSASparseSharedMemoryPayload:
        if not isinstance(raw, dict):
            raise TypeError("DSA Sparse shared-memory payload must be a dictionary")
        name = raw.get("name")
        size = raw.get("size")
        cache_kind = raw.get("cache_kind")
        cache_layer_name = raw.get("cache_layer_name")
        if not isinstance(name, str):
            raise TypeError("DSA Sparse shared-memory payload name must be a string")
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("DSA Sparse shared-memory payload size must be an integer")
        if not isinstance(cache_kind, str):
            raise TypeError("DSA Sparse shared-memory cache kind must be a string")
        if not isinstance(cache_layer_name, str):
            raise TypeError("DSA Sparse shared-memory cache layer name must be a string")
        raw_cache_planes = raw.get("cache_planes", ())
        raw_tail_planes = raw.get("tail_planes", ())
        if not isinstance(raw_cache_planes, (list, tuple)) or not isinstance(raw_tail_planes, (list, tuple)):
            raise TypeError("DSA Sparse shared-memory planes must be sequences")
        return cls(
            name=name,
            size=size,
            cache_kind=cache_kind,
            cache_layer_name=cache_layer_name,
            cache_planes=tuple(DSASparseSharedMemoryPlane.from_dict(plane) for plane in raw_cache_planes),
            tail_planes=tuple(DSASparseSharedMemoryPlane.from_dict(plane) for plane in raw_tail_planes),
        )


class DSASparseSharedMemoryReader:
    """Open one published payload and unlink it only after successful consume."""

    def __init__(
        self,
        payload: DSASparseSharedMemoryPayload,
        *,
        root: Path,
    ) -> None:
        self.payload = payload
        self._path = root / payload.name
        self._file = self._path.open("r+b", buffering=0)
        actual_size = os.fstat(self._file.fileno()).st_size
        if actual_size != payload.size:
            self._file.close()
            raise RuntimeError(
                "DSA Sparse shared-memory payload size changed before consume: "
                f"expected={payload.size}, actual={actual_size}."
            )
        self._mapping = mmap.mmap(
            self._file.fileno(),
            payload.size,
            access=mmap.ACCESS_WRITE,
        )

    def tensor(self, plane: DSASparseSharedMemoryPlane) -> torch.Tensor:
        # Detach the returned tensor from the mmap. This lets exception paths
        # close the mapping deterministically even when the caller still owns
        # the returned tensor or its traceback keeps a reference alive.
        byte_view = torch.frombuffer(
            self._mapping,
            dtype=torch.uint8,
            count=plane.nbytes,
            offset=plane.offset,
        )
        return byte_view.view(plane.torch_dtype).view(plane.shape).clone()

    def unlink(self) -> None:
        self._path.unlink(missing_ok=True)

    def close(self) -> None:
        self._mapping.close()
        self._file.close()

    def __enter__(self) -> DSASparseSharedMemoryReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class DSASparseSharedMemoryStore:
    """Publish and consume request-scoped payloads through ``/dev/shm``."""

    def __init__(
        self,
        root: str | os.PathLike[str] = DSA_SPARSE_SHARED_MEMORY_ROOT,
    ) -> None:
        self.root = Path(root)

    @staticmethod
    def _physical_block_ids(
        logical_block_ids: torch.Tensor,
        block_scale: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        logical = logical_block_ids.to(
            device=device,
            dtype=torch.int64,
        ).reshape(-1)
        offsets = torch.arange(block_scale, device=device, dtype=torch.int64)
        return (logical[:, None] * block_scale + offsets[None, :]).reshape(-1)

    def publish(
        self,
        *,
        cache_kind: str,
        cache_layer_name: str,
        cache: tuple[torch.Tensor, ...],
        cache_block_ids: torch.Tensor,
        logical_num_blocks: int,
        main_cache: tuple[torch.Tensor, ...] = (),
        main_tail_block_id: int | None = None,
        tail_valid_count: int = 0,
    ) -> DSASparseSharedMemoryPayload:
        tensors: list[tuple[str, torch.Tensor, int]] = []
        if not cache:
            raise ValueError("DSA Sparse shared-memory cache planes are required")
        if cache_kind not in DSA_SPARSE_SHARED_MEMORY_CACHE_KINDS:
            raise ValueError(f"DSA Sparse shared-memory cache kind is invalid: {cache_kind!r}")
        if not cache_layer_name:
            raise ValueError("DSA Sparse shared-memory cache has no layer name")
        if logical_num_blocks <= 0:
            raise ValueError("DSA Sparse shared-memory logical block count must be positive")
        for plane in cache:
            if plane.shape[0] % logical_num_blocks:
                raise ValueError("DSA Sparse shared cache physical blocks do not divide by the scheduler block count")
            block_scale = int(plane.shape[0]) // logical_num_blocks
            physical_ids = self._physical_block_ids(
                cache_block_ids,
                block_scale,
                device=plane.device,
            )
            compact = plane.index_select(0, physical_ids).detach().to(device="cpu").contiguous()
            tensors.append(("cache", compact, block_scale))

        if tail_valid_count:
            if main_tail_block_id is None:
                raise ValueError("DSA Sparse shared-memory tail has no source block")
            for plane in main_cache:
                compact = plane[main_tail_block_id, :tail_valid_count].detach().to(device="cpu").contiguous()
                tensors.append(("tail", compact, 1))

        total_size = sum(tensor.numel() * tensor.element_size() for _, tensor, _ in tensors)
        self.root.mkdir(parents=True, exist_ok=True)
        file_descriptor, path = tempfile.mkstemp(
            prefix=DSA_SPARSE_SHARED_MEMORY_PREFIX,
            dir=self.root,
        )
        os.fchmod(file_descriptor, 0o600)
        cache_planes: list[DSASparseSharedMemoryPlane] = []
        tail_planes: list[DSASparseSharedMemoryPlane] = []
        try:
            os.ftruncate(file_descriptor, total_size)
            with mmap.mmap(file_descriptor, total_size, access=mmap.ACCESS_WRITE) as mapping:
                offset = 0
                for kind, tensor, block_scale in tensors:
                    try:
                        dtype_name = _NAME_BY_DTYPE[tensor.dtype]
                    except KeyError as error:
                        raise ValueError(
                            f"Unsupported DSA Sparse shared-memory tensor dtype: {tensor.dtype}."
                        ) from error
                    byte_tensor = tensor.view(torch.uint8).reshape(-1)
                    nbytes = byte_tensor.numel()
                    mapping[offset : offset + nbytes] = byte_tensor.numpy().tobytes()
                    plane = DSASparseSharedMemoryPlane(
                        offset=offset,
                        nbytes=nbytes,
                        dtype=dtype_name,
                        shape=tuple(int(dim) for dim in tensor.shape),
                        block_scale=block_scale,
                    )
                    if kind == "cache":
                        cache_planes.append(plane)
                    else:
                        tail_planes.append(plane)
                    offset += nbytes
                mapping.flush()
        except BaseException:
            Path(path).unlink(missing_ok=True)
            raise
        finally:
            os.close(file_descriptor)

        return DSASparseSharedMemoryPayload(
            name=Path(path).name,
            size=total_size,
            cache_kind=cache_kind,
            cache_layer_name=cache_layer_name,
            cache_planes=tuple(cache_planes),
            tail_planes=tuple(tail_planes),
        )

    def open(
        self,
        payload: DSASparseSharedMemoryPayload,
    ) -> DSASparseSharedMemoryReader:
        return DSASparseSharedMemoryReader(payload, root=self.root)

    def unlink(self, payload: DSASparseSharedMemoryPayload) -> None:
        (self.root / payload.name).unlink(missing_ok=True)


__all__ = [
    "DSA_SPARSE_SHARED_MEMORY_PREFIX",
    "DSA_SPARSE_SHARED_MEMORY_CACHE_KINDS",
    "DSASparseSharedMemoryPayload",
    "DSASparseSharedMemoryPlane",
    "DSASparseSharedMemoryReader",
    "DSASparseSharedMemoryStore",
]
