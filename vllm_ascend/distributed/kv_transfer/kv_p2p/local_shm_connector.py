# SPDX-License-Identifier: Apache-2.0
"""Temporary same-host KV connector backed by file mappings in /dev/shm.

This connector intentionally supports one narrow deployment shape:

* Prefill and Decode run in the same container;
* Prefill TP equals Decode TP, and rank ``r`` copies to rank ``r``;
* DP, PP and context parallel sizes are one;
* copies are synchronous D2H/H2D operations.

The existing Mooncake scheduler implementation remains the control plane.  No
Mooncake TransferEngine, HCCL data channel, background thread, shared-memory
pool or TP resharding is used here.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.attention.dsa_sparse_pd import (
    DSASparsePDHandoff,
    get_dsa_sparse_pd_handoff,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    DSASparsePDWorkerMetadata,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    ReqMeta,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


_MANIFEST_VERSION = 1
_ALIGNMENT = 64


def _normalize_block_ids(block_ids: BlockIds) -> BlockIds:
    return tuple([int(block_id) for block_id in group] for group in block_ids)


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _cache_planes(cache: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(cache, torch.Tensor):
        return (cache,)
    if isinstance(cache, (tuple, list)):
        if not cache:
            raise ValueError("LocalShmConnector cache requires at least one tensor plane")
        if all(isinstance(plane, torch.Tensor) for plane in cache):
            return tuple(cache)
    raise TypeError(f"Unsupported KV cache type: {type(cache).__name__}")


@dataclass(frozen=True)
class LocalShmSendSpec:
    remote_block_ids: BlockIds
    dsa_sparse_handoff: DSASparsePDHandoff | None


class LocalShmConnectorMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, LocalShmSendSpec] = {}
        self.reqs_in_batch: set[str] = set()


class LocalShmConnectorScheduler(MooncakeConnectorScheduler):
    """Reuse Mooncake's request protocol while adding P-side block ids."""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._local_shm_send_specs: dict[str, LocalShmSendSpec] = {}
        logger.info("Initialized LocalShmConnector scheduler %s", engine_id)

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        delay_free, transfer_params = super().request_finished(request, block_ids)
        if delay_free:
            if transfer_params is None:
                raise RuntimeError("LocalShmConnector delayed KV blocks without transfer parameters")
            self._local_shm_send_specs[request.request_id] = LocalShmSendSpec(
                remote_block_ids=_normalize_block_ids(transfer_params["remote_block_ids"]),
                dsa_sparse_handoff=get_dsa_sparse_pd_handoff(transfer_params),
            )
        return delay_free, transfer_params

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        mooncake_meta = super().build_connector_meta(scheduler_output)
        if not isinstance(mooncake_meta, MooncakeConnectorMetadata):
            raise TypeError("Unexpected scheduler metadata type")

        metadata = LocalShmConnectorMetadata()
        metadata.requests = mooncake_meta.requests
        metadata.reqs_in_batch = mooncake_meta.reqs_in_batch
        for request_id in mooncake_meta.requests_to_send:
            try:
                metadata.requests_to_send[request_id] = self._local_shm_send_specs.pop(request_id)
            except KeyError as error:
                raise RuntimeError(
                    f"LocalShmConnector has no source block metadata for request {request_id!r}"
                ) from error
        return metadata


@dataclass
class _PayloadRecord:
    manifest: dict[str, Any]
    plane: torch.Tensor
    source_block_ids: tuple[int, ...]


class LocalShmConnectorWorker:
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        init_ascend_config(vllm_config)
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self.kv_cache_config = kv_cache_config
        self.num_blocks = int(kv_cache_config.num_blocks)
        self.block_size = int(vllm_config.cache_config.block_size)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.ascend_config = get_ascend_config()
        self.dsa_sparse_config = self.ascend_config.dsa_sparse_config
        self.total_layers = vllm_config.model_config.get_total_num_hidden_layers()

        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        prefill_config = extra_config.get("prefill", {})
        decode_config = extra_config.get("decode", {})
        prefill_tp_size = int(prefill_config.get("tp_size", 0))
        decode_tp_size = int(decode_config.get("tp_size", 0))
        if prefill_tp_size <= 0 or decode_tp_size <= 0:
            raise ValueError("LocalShmConnector requires prefill.tp_size and decode.tp_size")
        if prefill_tp_size != decode_tp_size or self.tp_size != prefill_tp_size:
            raise ValueError(
                "LocalShmConnector requires equal Prefill and Decode TP sizes "
                f"(prefill={prefill_tp_size}, decode={decode_tp_size}, local={self.tp_size})"
            )
        if int(prefill_config.get("dp_size", 1)) != 1 or int(decode_config.get("dp_size", 1)) != 1:
            raise ValueError("LocalShmConnector does not support data parallelism")
        parallel_config = vllm_config.parallel_config
        if parallel_config.pipeline_parallel_size != 1:
            raise ValueError("LocalShmConnector does not support pipeline parallelism")
        if parallel_config.prefill_context_parallel_size != 1 or parallel_config.decode_context_parallel_size != 1:
            raise ValueError("LocalShmConnector does not support context parallelism")
        if self.kv_role not in ("kv_producer", "kv_consumer"):
            raise ValueError("LocalShmConnector supports only kv_producer or kv_consumer")

        shm_dir = Path(str(extra_config.get("shm_dir", "/dev/shm/vllm-ascend-local-kv")))
        if not shm_dir.is_absolute():
            raise ValueError("LocalShmConnector shm_dir must be an absolute path")
        namespace = str(extra_config.get("shm_namespace", "default"))
        namespace_key = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:20]
        self.shm_dir = shm_dir / namespace_key
        self.shm_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(extra_config.get("shm_timeout", 120.0))
        if self.timeout <= 0:
            raise ValueError("LocalShmConnector shm_timeout must be positive")

        self.kv_caches: dict[str, tuple[torch.Tensor, ...]] = {}
        self._layer_group_ids: dict[str, int] = {}
        self._group_tokens_per_block: dict[int, int] = {}
        self._dsa_sparse_aux_caches: dict[str, tuple[torch.Tensor, ...]] = {}
        self._dsa_sparse_tail_layouts: dict[str, tuple[int, int]] = {}
        self._dsa_sparse_request_rows: dict[str, int] = {}
        self._dsa_sparse_layer_topk_by_request: dict[str, dict[str, list[int]]] = {}
        self._dsa_sparse_partial_tail_blocks_by_request: dict[str, dict[str, int]] = {}
        self._finished_sending: set[str] = set()
        self._finished_receiving: set[str] = set()
        self.xfer_handshake_metadata = None
        logger.info(
            "Initialized LocalShmConnector worker: role=%s tp_rank=%d tp_size=%d directory=%s",
            self.kv_role,
            self.tp_rank,
            self.tp_size,
            self.shm_dir,
        )

    def _request_paths(self, engine_id: str, request_id: str) -> tuple[Path, Path]:
        identity = f"{engine_id}\0{request_id}\0{self.tp_rank}".encode()
        key = hashlib.sha256(identity).hexdigest()
        return self.shm_dir / f"{key}.data", self.shm_dir / f"{key}.ready.json"

    def _transfer_layer_names(self) -> set[str] | None:
        if self.dsa_sparse_config is None:
            return None

        from vllm.v1.worker.utils import extract_layer_index

        num_attn_module = 2 if self.vllm_config.model_config.hf_text_config.model_type == "longcat_flash" else 1
        transfer_layers: set[str] = set()
        for group in self.kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                layer_spec = group.kv_cache_spec
                if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                    layer_spec = layer_spec.kv_cache_specs[layer_name]
                is_target_main = False
                if isinstance(layer_spec, MLAAttentionSpec) and "mtp" not in layer_name:
                    layer_index = extract_layer_index(layer_name, num_attn_module)
                    is_target_main = layer_index < self.total_layers
                if not is_target_main:
                    transfer_layers.add(layer_name)
        return transfer_layers

    @staticmethod
    def _compress_ratio(group: Any) -> int:
        specs = tuple(
            group.kv_cache_spec.kv_cache_specs.values()
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else (group.kv_cache_spec,)
        )
        return max([1, *(int(getattr(spec, "compress_ratio", 1)) for spec in specs)])

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        transfer_layers = self._transfer_layer_names()
        self.kv_caches = {
            layer_name: _cache_planes(cache)
            for layer_name, cache in kv_caches.items()
            if transfer_layers is None or layer_name in transfer_layers
        }
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
                first_spec = next(iter(group.kv_cache_spec.kv_cache_specs.values()))
            else:
                first_spec = group.kv_cache_spec
            block_size = int(
                getattr(group.kv_cache_spec, "block_size", getattr(first_spec, "block_size", self.block_size))
            )
            self._group_tokens_per_block[group_id] = block_size * self._compress_ratio(group)
            for layer_name in group.layer_names:
                if layer_name in self.kv_caches:
                    self._layer_group_ids[layer_name] = group_id
        missing_groups = sorted(set(self.kv_caches) - set(self._layer_group_ids))
        if missing_groups:
            raise RuntimeError(f"LocalShmConnector could not map cache layers to groups: {missing_groups}")

    def register_dsa_sparse_aux_caches(
        self,
        caches: dict[str, tuple[torch.Tensor, ...]],
        tail_layouts: dict[str, tuple[int, int]],
    ) -> None:
        if set(tail_layouts) - set(caches):
            raise ValueError("DSA Sparse tail layouts reference unknown cache layers")
        self._dsa_sparse_aux_caches = {
            layer_name: _cache_planes(cache) for layer_name, cache in caches.items()
        }
        self._dsa_sparse_tail_layouts = {
            layer_name: (int(layout[0]), int(layout[1])) for layer_name, layout in tail_layouts.items()
        }

    def set_dsa_sparse_request_rows(self, request_rows: Mapping[str, int]) -> None:
        self._dsa_sparse_request_rows = {
            str(request_id): int(request_row) for request_id, request_row in request_rows.items()
        }

    def capture_dsa_sparse_layer_topk(
        self,
        layer_name: str,
        attn_metadata: "AttentionMetadata",
    ) -> None:
        if getattr(self.dsa_sparse_config, "kv_role", None) != "kv_producer":
            return
        layer_metadata = attn_metadata.get(layer_name) if isinstance(attn_metadata, Mapping) else attn_metadata
        producer_context = getattr(layer_metadata, "dsa_sparse_producer_context", None)
        if producer_context is None:
            return
        for request_id, token_ids in producer_context.layer_topk(layer_name).items():
            layers = self._dsa_sparse_layer_topk_by_request.setdefault(request_id, {})
            normalized = [int(token_id) for token_id in token_ids]
            current = layers.get(layer_name)
            if current is not None and current != normalized:
                raise RuntimeError(
                    f"DSA Sparse P worker captured conflicting TopK: request={request_id!r}, "
                    f"layer={layer_name!r}"
                )
            layers[layer_name] = normalized
        for request_id, block_id in producer_context.layer_partial_tail_blocks(layer_name).items():
            layers = self._dsa_sparse_partial_tail_blocks_by_request.setdefault(request_id, {})
            layers[layer_name] = int(block_id)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        if not self._dsa_sparse_layer_topk_by_request:
            return None
        topk = self._dsa_sparse_layer_topk_by_request
        partial_tail = self._dsa_sparse_partial_tail_blocks_by_request
        self._dsa_sparse_layer_topk_by_request = {}
        self._dsa_sparse_partial_tail_blocks_by_request = {}
        return DSASparsePDWorkerMetadata(
            {
                request_id: {
                    self.tp_rank: {
                        layer_name: list(token_ids) for layer_name, token_ids in layers.items()
                    }
                }
                for request_id, layers in topk.items()
            },
            {
                request_id: {
                    self.tp_rank: {
                        layer_name: int(block_id) for layer_name, block_id in layers.items()
                    }
                }
                for request_id, layers in partial_tail.items()
            },
        )

    def _normal_payload_records(self, send_spec: LocalShmSendSpec) -> list[_PayloadRecord]:
        records: list[_PayloadRecord] = []
        for layer_name, planes in self.kv_caches.items():
            group_id = self._layer_group_ids[layer_name]
            source_block_ids = tuple(send_spec.remote_block_ids[group_id])
            if not source_block_ids:
                continue
            for plane_index, plane in enumerate(planes):
                if plane.ndim < 2 or not plane.is_contiguous():
                    raise ValueError(f"LocalShmConnector requires contiguous paged cache: {layer_name}")
                if plane.shape[0] % self.num_blocks:
                    raise ValueError(f"LocalShmConnector cache block scale is not integral: {layer_name}")
                block_scale = int(plane.shape[0] // self.num_blocks)
                if block_scale <= 0:
                    raise ValueError(f"LocalShmConnector cache has no physical blocks: {layer_name}")
                if any(block_id < 0 or block_id >= self.num_blocks for block_id in source_block_ids):
                    raise IndexError(f"LocalShmConnector source block is out of range: {layer_name}")
                payload_shape = [len(source_block_ids) * block_scale, *map(int, plane.shape[1:])]
                records.append(
                    _PayloadRecord(
                        manifest={
                            "kind": "kv",
                            "group_id": group_id,
                            "layer_name": layer_name,
                            "plane_index": plane_index,
                            "dtype": str(plane.dtype),
                            "source_block_ids": list(source_block_ids),
                            "block_scale": block_scale,
                            "payload_shape": payload_shape,
                        },
                        plane=plane,
                        source_block_ids=source_block_ids,
                    )
                )
        return records

    def _tail_payload_records(self, send_spec: LocalShmSendSpec) -> list[_PayloadRecord]:
        handoff = send_spec.dsa_sparse_handoff
        if handoff is None:
            return []
        valid_count = handoff.stored_token_count % handoff.block_size
        if valid_count == 0:
            return []
        if handoff.block_size != self.block_size:
            raise ValueError("LocalShmConnector partial-tail block size differs from the local cache")
        source_layers = handoff.partial_tail_blocks_by_rank.get(self.tp_rank, {})
        if set(source_layers) != set(self._dsa_sparse_aux_caches):
            raise RuntimeError(
                "LocalShmConnector partial-tail layers do not match the registered Prefill Main caches"
            )

        records: list[_PayloadRecord] = []
        for layer_name, source_block_id in source_layers.items():
            for plane_index, plane in enumerate(self._dsa_sparse_aux_caches[layer_name]):
                if plane.ndim < 2 or plane.shape[1] != self.block_size or not plane.is_contiguous():
                    raise ValueError(f"LocalShmConnector has unsupported partial-tail cache: {layer_name}")
                if source_block_id < 0 or source_block_id >= plane.shape[0]:
                    raise IndexError(f"LocalShmConnector partial-tail source block is out of range: {layer_name}")
                payload_shape = [valid_count, *map(int, plane.shape[2:])]
                records.append(
                    _PayloadRecord(
                        manifest={
                            "kind": "tail",
                            "layer_name": layer_name,
                            "plane_index": plane_index,
                            "dtype": str(plane.dtype),
                            "source_block_id": int(source_block_id),
                            "valid_count": valid_count,
                            "payload_shape": payload_shape,
                        },
                        plane=plane,
                        source_block_ids=(int(source_block_id),),
                    )
                )
        return records

    @staticmethod
    def _record_nbytes(record: _PayloadRecord) -> int:
        elements = 1
        for dimension in record.manifest["payload_shape"]:
            elements *= int(dimension)
        return elements * record.plane.element_size()

    @staticmethod
    def _mapped_tensor(mapping: mmap.mmap, record: dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
        raw = torch.frombuffer(
            mapping,
            dtype=torch.uint8,
            count=int(record["nbytes"]),
            offset=int(record["offset"]),
        )
        return raw.view(dtype).reshape(record["payload_shape"])

    def _publish(self, request_id: str, send_spec: LocalShmSendSpec) -> None:
        data_path, ready_path = self._request_paths(self.engine_id, request_id)
        if data_path.exists() or ready_path.exists():
            raise FileExistsError(f"LocalShmConnector request already exists: {request_id!r}")

        records = self._normal_payload_records(send_spec) + self._tail_payload_records(send_spec)
        total_bytes = 0
        for record in records:
            total_bytes = _align_up(total_bytes)
            record.manifest["offset"] = total_bytes
            record.manifest["nbytes"] = self._record_nbytes(record)
            total_bytes += int(record.manifest["nbytes"])
        mapped_size = max(total_bytes, 1)

        temporary_data = data_path.with_name(f"{data_path.name}.tmp.{os.getpid()}")
        temporary_ready = ready_path.with_name(f"{ready_path.name}.tmp.{os.getpid()}")
        try:
            with temporary_data.open("w+b") as data_file:
                data_file.truncate(mapped_size)
                with mmap.mmap(data_file.fileno(), mapped_size, access=mmap.ACCESS_WRITE) as mapping:
                    for payload_record in records:
                        manifest_record = payload_record.manifest
                        mapped = self._mapped_tensor(mapping, manifest_record, payload_record.plane.dtype)
                        if manifest_record["kind"] == "kv":
                            scale = int(manifest_record["block_scale"])
                            for position, source_block_id in enumerate(payload_record.source_block_ids):
                                source_begin = source_block_id * scale
                                mapped[position * scale : (position + 1) * scale].copy_(
                                    payload_record.plane[source_begin : source_begin + scale],
                                    non_blocking=False,
                                )
                        else:
                            source_block_id = payload_record.source_block_ids[0]
                            valid_count = int(manifest_record["valid_count"])
                            mapped.copy_(
                                payload_record.plane[source_block_id, :valid_count],
                                non_blocking=False,
                            )
                        del mapped
                    mapping.flush()
            os.replace(temporary_data, data_path)

            manifest = {
                "version": _MANIFEST_VERSION,
                "engine_id": self.engine_id,
                "request_id": request_id,
                "tp_rank": self.tp_rank,
                "tp_size": self.tp_size,
                "block_size": self.block_size,
                "data_size": mapped_size,
                "records": [record.manifest for record in records],
            }
            with temporary_ready.open("w", encoding="utf-8") as ready_file:
                json.dump(manifest, ready_file, sort_keys=True)
                ready_file.flush()
                os.fsync(ready_file.fileno())
            os.replace(temporary_ready, ready_path)
        except BaseException:
            temporary_data.unlink(missing_ok=True)
            temporary_ready.unlink(missing_ok=True)
            raise

        logger.info(
            "LocalShmConnector published request=%s rank=%d records=%d bytes=%d",
            request_id,
            self.tp_rank,
            len(records),
            total_bytes,
        )

    def _wait_for_manifest(self, ready_path: Path, request_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                with ready_path.open(encoding="utf-8") as ready_file:
                    return json.load(ready_file)
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"LocalShmConnector timed out waiting for request {request_id!r}, rank {self.tp_rank}"
                    ) from None
                time.sleep(0.01)

    def _validate_manifest(self, manifest: dict[str, Any], meta: ReqMeta) -> None:
        expected = {
            "version": _MANIFEST_VERSION,
            "engine_id": meta.remote_engine_id,
            "request_id": meta.remote_request_id,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "block_size": meta.remote_block_size or self.block_size,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"LocalShmConnector manifest does not match the Decode request: {mismatches}")

    def _validate_payload_plane_layouts(self, manifest: dict[str, Any], meta: ReqMeta) -> None:
        records = manifest.get("records")
        if not isinstance(records, list):
            raise RuntimeError("LocalShmConnector manifest records must be a list")

        expected: dict[tuple[str, str], set[int]] = {}
        for layer_name, planes in self.kv_caches.items():
            group_id = self._layer_group_ids[layer_name]
            if group_id >= len(meta.remote_block_ids):
                raise IndexError(f"LocalShmConnector group id is out of range: {group_id}")
            if meta.remote_block_ids[group_id]:
                expected[("kv", layer_name)] = set(range(len(planes)))

        handoff = meta.dsa_sparse_handoff
        if handoff is not None and handoff.stored_token_count % handoff.block_size:
            source_layers = handoff.partial_tail_blocks_by_rank.get(self.tp_rank, {})
            if set(source_layers) != set(self._dsa_sparse_aux_caches):
                raise RuntimeError(
                    "LocalShmConnector partial-tail layers do not match the registered Decode Main caches"
                )
            missing_layouts = set(source_layers) - set(self._dsa_sparse_tail_layouts)
            if missing_layouts:
                raise RuntimeError(
                    "LocalShmConnector Decode partial-tail layouts are missing layers: "
                    f"{sorted(missing_layouts)}"
                )
            for layer_name in source_layers:
                expected[("tail", layer_name)] = set(
                    range(len(self._dsa_sparse_aux_caches[layer_name]))
                )

        observed: dict[tuple[str, str], list[int]] = {}
        for record in records:
            kind = str(record.get("kind"))
            layer_name = str(record.get("layer_name"))
            plane_index = int(record["plane_index"])
            observed.setdefault((kind, layer_name), []).append(plane_index)

        mismatches = {
            f"{kind}:{layer_name}": {
                "published": sorted(observed.get((kind, layer_name), [])),
                "expected": sorted(expected.get((kind, layer_name), set())),
            }
            for kind, layer_name in sorted(set(expected) | set(observed))
            if sorted(observed.get((kind, layer_name), []))
            != sorted(expected.get((kind, layer_name), set()))
        }
        if mismatches:
            raise RuntimeError(
                "LocalShmConnector payload cache plane layouts do not match Decode: "
                f"{mismatches}"
            )

    def _load_kv_record(self, mapping: mmap.mmap, record: dict[str, Any], meta: ReqMeta) -> None:
        group_id = int(record["group_id"])
        layer_name = str(record["layer_name"])
        plane_index = int(record["plane_index"])
        if layer_name not in self.kv_caches or self._layer_group_ids.get(layer_name) != group_id:
            raise RuntimeError(f"LocalShmConnector Decode cache does not contain {layer_name!r}")
        if group_id >= len(meta.local_block_ids) or group_id >= len(meta.remote_block_ids):
            raise IndexError(f"LocalShmConnector group id is out of range: {group_id}")

        source_block_ids = [int(block_id) for block_id in record["source_block_ids"]]
        expected_remote_ids = [int(block_id) for block_id in meta.remote_block_ids[group_id]]
        if source_block_ids != expected_remote_ids:
            raise RuntimeError(f"LocalShmConnector source block ids differ for group {group_id}")
        tokens_per_block = self._group_tokens_per_block[group_id]
        if meta.num_computed_tokens % tokens_per_block:
            raise RuntimeError(
                "LocalShmConnector does not support a prefix ending inside a scheduler KV block"
            )
        source_start = meta.num_computed_tokens // tokens_per_block
        local_block_ids = [int(block_id) for block_id in meta.local_block_ids[group_id]]
        copy_count = min(len(local_block_ids), len(source_block_ids) - source_start)
        if copy_count < 0:
            raise RuntimeError("LocalShmConnector prefix exceeds the published KV payload")

        planes = self.kv_caches[layer_name]
        if plane_index < 0 or plane_index >= len(planes):
            raise IndexError(
                f"LocalShmConnector Decode cache plane is out of range: {layer_name}[{plane_index}]"
            )
        plane = planes[plane_index]
        if str(plane.dtype) != record["dtype"]:
            raise TypeError(f"LocalShmConnector dtype differs for {layer_name!r}")
        if plane.shape[0] % self.num_blocks:
            raise ValueError(f"LocalShmConnector Decode cache block scale is not integral: {layer_name}")
        block_scale = int(plane.shape[0] // self.num_blocks)
        if block_scale != int(record["block_scale"]):
            raise ValueError(f"LocalShmConnector block scale differs for {layer_name!r}")
        mapped = self._mapped_tensor(mapping, record, plane.dtype)
        expected_shape = (len(source_block_ids) * block_scale, *map(int, plane.shape[1:]))
        if tuple(mapped.shape) != expected_shape:
            raise ValueError(f"LocalShmConnector payload shape differs for {layer_name!r}")
        for position in range(copy_count):
            destination_block = local_block_ids[position]
            if destination_block < 0 or destination_block >= self.num_blocks:
                raise IndexError(f"LocalShmConnector destination block is out of range: {layer_name}")
            source_position = source_start + position
            plane[destination_block * block_scale : (destination_block + 1) * block_scale].copy_(
                mapped[source_position * block_scale : (source_position + 1) * block_scale],
                non_blocking=False,
            )
        del mapped

    def _load_tail_record(
        self,
        mapping: mmap.mmap,
        record: dict[str, Any],
        request_id: str,
        meta: ReqMeta,
    ) -> None:
        handoff = meta.dsa_sparse_handoff
        if handoff is None:
            raise RuntimeError("LocalShmConnector received a partial tail without DSA Sparse handoff")
        request_row = self._dsa_sparse_request_rows.get(request_id)
        if request_row is None:
            raise RuntimeError("DSA Sparse partial-tail handoff has no Decode request row")
        layer_name = str(record["layer_name"])
        plane_index = int(record["plane_index"])
        if layer_name not in self._dsa_sparse_aux_caches or layer_name not in self._dsa_sparse_tail_layouts:
            raise RuntimeError(f"LocalShmConnector Decode Hot Cache does not contain {layer_name!r}")
        source_layers = handoff.partial_tail_blocks_by_rank.get(self.tp_rank, {})
        if int(record["source_block_id"]) != int(source_layers.get(layer_name, -1)):
            raise RuntimeError(f"LocalShmConnector partial-tail source differs for {layer_name!r}")
        valid_count = handoff.stored_token_count % handoff.block_size
        if valid_count != int(record["valid_count"]):
            raise ValueError(f"LocalShmConnector partial-tail length differs for {layer_name!r}")

        blocks_per_request, tail_block_offset = self._dsa_sparse_tail_layouts[layer_name]
        destination_block = request_row * blocks_per_request + tail_block_offset
        planes = self._dsa_sparse_aux_caches[layer_name]
        if plane_index < 0 or plane_index >= len(planes):
            raise IndexError(
                "LocalShmConnector Decode Hot Cache plane is out of range: "
                f"{layer_name}[{plane_index}]"
            )
        plane = planes[plane_index]
        if destination_block < 0 or destination_block >= plane.shape[0]:
            raise IndexError(f"LocalShmConnector partial-tail destination is out of range: {layer_name}")
        if str(plane.dtype) != record["dtype"]:
            raise TypeError(f"LocalShmConnector partial-tail dtype differs for {layer_name!r}")
        mapped = self._mapped_tensor(mapping, record, plane.dtype)
        expected_shape = (valid_count, *map(int, plane.shape[2:]))
        if tuple(mapped.shape) != expected_shape:
            raise ValueError(f"LocalShmConnector partial-tail shape differs for {layer_name!r}")
        plane[destination_block, :valid_count].copy_(mapped, non_blocking=False)
        del mapped

    def _load(self, request_id: str, meta: ReqMeta) -> None:
        data_path, ready_path = self._request_paths(meta.remote_engine_id, meta.remote_request_id)
        manifest = self._wait_for_manifest(ready_path, request_id)
        self._validate_manifest(manifest, meta)
        self._validate_payload_plane_layouts(manifest, meta)
        data_size = int(manifest["data_size"])
        success = False
        with data_path.open("r+b") as data_file:
            with mmap.mmap(data_file.fileno(), data_size, access=mmap.ACCESS_WRITE) as mapping:
                for record in manifest["records"]:
                    if record["kind"] == "kv":
                        self._load_kv_record(mapping, record, meta)
                    elif record["kind"] == "tail":
                        self._load_tail_record(mapping, record, request_id, meta)
                    else:
                        raise ValueError(f"Unknown LocalShmConnector payload kind: {record['kind']!r}")
                success = True
        if success:
            ready_path.unlink(missing_ok=True)
            data_path.unlink(missing_ok=True)
        logger.info(
            "LocalShmConnector loaded request=%s remote_request=%s rank=%d records=%d",
            request_id,
            meta.remote_request_id,
            self.tp_rank,
            len(manifest["records"]),
        )

    def start_load_kv(self, metadata: LocalShmConnectorMetadata) -> None:
        if self.kv_role == "kv_producer":
            for request_id, send_spec in metadata.requests_to_send.items():
                self._publish(request_id, send_spec)
                self._finished_sending.add(request_id)
            return
        for request_id, meta in metadata.requests.items():
            self._load(request_id, meta)
            self._finished_receiving.add(request_id)

    def get_finished(self) -> tuple[set[str], set[str]]:
        sending = self._finished_sending
        receiving = self._finished_receiving
        self._finished_sending = set()
        self._finished_receiving = set()
        return sending, receiving

    @staticmethod
    def get_block_ids_with_load_errors() -> set[int]:
        return set()


class LocalShmConnector(MooncakeConnector):
    """Mooncake-compatible control facade with a local mmap worker."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        if kv_cache_config is None:
            raise ValueError("LocalShmConnector requires kv_cache_config")
        engine_id = str(vllm_config.kv_transfer_config.engine_id)
        self.engine_id = engine_id
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: LocalShmConnectorScheduler | None = LocalShmConnectorScheduler(
                vllm_config, engine_id, kv_cache_config
            )
            self.connector_worker: LocalShmConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = LocalShmConnectorWorker(vllm_config, engine_id, kv_cache_config)
        else:
            raise ValueError(f"Unsupported LocalShmConnector role: {role}")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LocalShmConnectorMetadata):
            raise TypeError("LocalShmConnector received incompatible scheduler metadata")
        self.connector_worker.start_load_kv(metadata)
