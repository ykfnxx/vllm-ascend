"""Mooncake initial P/D handoff for DSA sparse offload.

This adapter deliberately does not use ``MooncakeConnector``'s HMA cache
registration or group transfer path.  DSA has two cache groups with different
physical block counts, so the adapter registers the real Indexer/MLA tensors
with Mooncake's TransferEngine and builds an explicit token-address plan:

* the complete Indexer history is copied into D's Indexer block table;
* each layer's selected resident MLA tokens are copied into slots [0, 8192);
* the partial dense tail is copied behind the 2048 free lookup slots.

KVIO remains the persistent backing store and services later lookup misses.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket

from vllm_ascend.distributed.kv_transfer.dsa_kvio_connector import (
    DSAKVIOConnector,
    DSAKVIOConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    DONE_RECVING_MSG,
    GET_META_MSG,
    KVCacheSendingThread,
    MooncakeAgentMetadata,
    MooncakeConnectorWorker,
    ensure_zmq_recv,
    ensure_zmq_send,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)
from vllm_ascend.dsa_sparse.dsa_pd import (
    DSA_KVIO_PD_MANIFEST_KEY,
    DSA_PD_INITIAL_TRANSPORT_KEY,
    DSA_PD_INITIAL_TRANSPORT_MOONCAKE,
    DSAKVIOPDManifest,
    DSAKVIOPDRequest,
    build_pd_resident_token_ids,
)
from vllm_ascend.dsa_sparse.dsa_spec_utils import (
    is_dsa_indexer_spec,
    is_dsa_mla_resident_spec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger("vllm.dsa_sparse")

_INDEXER_COMPONENT = "indexer"
_NOPE_COMPONENT = "nope"
_ROPE_COMPONENT = "rope"
_MOONCAKE_TRANSFER_BATCH_SIZE = 4096


@dataclass(frozen=True)
class _DSAMooncakeCacheRegion:
    layer_id: int
    component: str
    tensor: torch.Tensor
    num_blocks: int
    block_bytes: int
    token_bytes: int


@dataclass(frozen=True)
class _DSAMooncakeTransfer:
    local_address: int
    remote_address: int
    length: int


@dataclass
class DSAMooncakeConnectorMetadata(DSAKVIOConnectorMetadata):
    """Scheduler-to-worker requests plus P-side delayed-free state."""

    requests_to_send: dict[str, float] = field(default_factory=dict)
    reqs_in_batch: set[str] = field(default_factory=set)


def _layer_id(layer_name: str) -> int:
    try:
        return int(layer_name.split(".")[2])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Cannot extract DSA layer id from cache name {layer_name!r}"
        ) from exc


def _append_coalesced_transfer(
    transfers: list[_DSAMooncakeTransfer],
    *,
    local_address: int,
    remote_address: int,
    length: int,
) -> None:
    length = int(length)
    if length <= 0:
        return
    local_address = int(local_address)
    remote_address = int(remote_address)
    if transfers:
        previous = transfers[-1]
        if (
            previous.local_address + previous.length == local_address
            and previous.remote_address + previous.length == remote_address
        ):
            transfers[-1] = _DSAMooncakeTransfer(
                local_address=previous.local_address,
                remote_address=previous.remote_address,
                length=previous.length + length,
            )
            return
    transfers.append(
        _DSAMooncakeTransfer(
            local_address=local_address,
            remote_address=remote_address,
            length=length,
        )
    )


def _append_token_mapping(
    transfers: list[_DSAMooncakeTransfer],
    *,
    local_region: _DSAMooncakeCacheRegion,
    remote_base_address: int,
    remote_block_ids: list[int],
    local_block_ids: list[int],
    source_token_ids: list[int],
    destination_slots: list[int],
    block_size: int,
) -> None:
    if len(source_token_ids) != len(destination_slots):
        raise ValueError(
            "DSA Mooncake source-token and destination-slot counts differ"
        )
    region_transfers: list[_DSAMooncakeTransfer] = []
    for source_token, destination_slot in zip(
        source_token_ids, destination_slots, strict=True
    ):
        source_token = int(source_token)
        destination_slot = int(destination_slot)
        if source_token < 0 or destination_slot < 0:
            raise ValueError("DSA Mooncake token mappings cannot be negative")
        remote_logical_block, remote_offset = divmod(
            source_token, block_size
        )
        local_logical_block, local_offset = divmod(
            destination_slot, block_size
        )
        if remote_logical_block >= len(remote_block_ids):
            raise RuntimeError(
                "DSA Mooncake remote block table is too short for token "
                f"{source_token}"
            )
        if local_logical_block >= len(local_block_ids):
            raise RuntimeError(
                "DSA Mooncake local block table is too short for slot "
                f"{destination_slot}"
            )
        remote_address = (
            int(remote_base_address)
            + int(remote_block_ids[remote_logical_block])
            * local_region.block_bytes
            + remote_offset * local_region.token_bytes
        )
        local_address = (
            int(local_region.tensor.data_ptr())
            + int(local_block_ids[local_logical_block])
            * local_region.block_bytes
            + local_offset * local_region.token_bytes
        )
        _append_coalesced_transfer(
            region_transfers,
            local_address=local_address,
            remote_address=remote_address,
            length=local_region.token_bytes,
        )
    transfers.extend(region_transfers)


def _append_contiguous_range_mapping(
    transfers: list[_DSAMooncakeTransfer],
    *,
    local_region: _DSAMooncakeCacheRegion,
    remote_base_address: int,
    remote_block_ids: list[int],
    local_block_ids: list[int],
    source_token_start: int,
    destination_slot_start: int,
    token_count: int,
    block_size: int,
) -> None:
    source_token = int(source_token_start)
    destination_slot = int(destination_slot_start)
    remaining = int(token_count)
    region_transfers: list[_DSAMooncakeTransfer] = []
    while remaining > 0:
        remote_logical_block, remote_offset = divmod(
            source_token, block_size
        )
        local_logical_block, local_offset = divmod(
            destination_slot, block_size
        )
        if remote_logical_block >= len(remote_block_ids):
            raise RuntimeError(
                "DSA Mooncake remote block table is too short for token "
                f"{source_token}"
            )
        if local_logical_block >= len(local_block_ids):
            raise RuntimeError(
                "DSA Mooncake local block table is too short for slot "
                f"{destination_slot}"
            )
        chunk_tokens = min(
            remaining,
            block_size - remote_offset,
            block_size - local_offset,
        )
        _append_coalesced_transfer(
            region_transfers,
            local_address=(
                int(local_region.tensor.data_ptr())
                + int(local_block_ids[local_logical_block])
                * local_region.block_bytes
                + local_offset * local_region.token_bytes
            ),
            remote_address=(
                int(remote_base_address)
                + int(remote_block_ids[remote_logical_block])
                * local_region.block_bytes
                + remote_offset * local_region.token_bytes
            ),
            length=chunk_tokens * local_region.token_bytes,
        )
        source_token += chunk_tokens
        destination_slot += chunk_tokens
        remaining -= chunk_tokens
    transfers.extend(region_transfers)


class DSAMooncakeConnector(DSAKVIOConnector):
    """Use Mooncake for initial P-to-D materialization and KVIO afterward."""

    _initial_transport = DSA_PD_INITIAL_TRANSPORT_MOONCAKE

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._connector_metadata = DSAMooncakeConnectorMetadata()
        self._requests_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()
        self._multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}

        parallel_config = vllm_config.parallel_config
        self._side_channel_host = get_ip()
        pcp_size = int(
            getattr(parallel_config, "prefill_context_parallel_size", 1)
        )
        self._side_channel_port = (
            int(vllm_config.kv_transfer_config.kv_port)
            + int(getattr(parallel_config, "data_parallel_rank", 0))
            * int(parallel_config.tensor_parallel_size)
            * int(parallel_config.pipeline_parallel_size)
            * pcp_size
        )
        self._worker: DSAMooncakeConnectorWorker | None = None
        if role == KVConnectorRole.WORKER:
            if kv_cache_config is None:
                raise ValueError(
                    "DSAMooncakeConnector worker requires KV cache config"
                )
            self._worker = DSAMooncakeConnectorWorker(
                vllm_config,
                str(self._engine_id),
                kv_cache_config,
            )

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        params = request.kv_transfer_params
        if isinstance(params, dict) and (
            params.get("do_remote_prefill", False)
            or params.get("do_remote_decode", False)
        ):
            self._reqs_in_batch.add(request.request_id)
        super().update_state_after_alloc(
            request, blocks, num_external_tokens
        )
        if not self._is_consumer or int(num_external_tokens) <= 0:
            return
        pd_request = self._requests_need_load.get(request.request_id)
        if pd_request is None:
            return
        if not isinstance(params, dict):
            raise RuntimeError("DSA Mooncake allocation has no route params")
        required = (
            "remote_block_ids",
            "remote_engine_id",
            "remote_request_id",
            "remote_host",
            "remote_port",
        )
        missing = [name for name in required if name not in params]
        if missing:
            raise ValueError(
                "DSA Mooncake handoff is missing route fields: "
                f"{missing}"
            )
        remote_group_block_ids = params["remote_block_ids"]
        if (
            not isinstance(remote_group_block_ids, (list, tuple))
            or self._indexer_group_id is None
            or self._resident_group_id is None
            or len(remote_group_block_ids)
            <= max(self._indexer_group_id, self._resident_group_id)
        ):
            raise ValueError(
                "DSA Mooncake handoff has invalid cache-group block tables"
            )
        self._requests_need_load[request.request_id] = replace(
            pd_request,
            initial_transport=DSA_PD_INITIAL_TRANSPORT_MOONCAKE,
            remote_indexer_block_ids=[
                int(block_id)
                for block_id in remote_group_block_ids[
                    self._indexer_group_id
                ]
            ],
            remote_resident_block_ids=[
                int(block_id)
                for block_id in remote_group_block_ids[
                    self._resident_group_id
                ]
            ],
            remote_engine_id=str(params["remote_engine_id"]),
            remote_request_id=str(params["remote_request_id"]),
            remote_host=str(params["remote_host"]),
            remote_port=int(params["remote_port"]),
            remote_multi_nodes_meta_mapping=dict(
                params.get("remote_multi_nodes_meta_mapping", {})
            ),
        )
        # Prevent a resumed scheduler pass from scheduling the same transfer.
        params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        base_metadata = super().build_connector_meta(scheduler_output)
        metadata = DSAMooncakeConnectorMetadata(
            dsa_requests=list(base_metadata.dsa_requests),
            requests_to_send=self._requests_need_send,
            reqs_in_batch=self._reqs_in_batch,
        )
        self._requests_need_send = {}
        self._reqs_in_batch = set()
        return metadata

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._is_producer:
            raise RuntimeError(
                "DSAMooncakeConnector requires all DSA cache-group block "
                "tables when finishing a P request"
            )
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._requests_need_load.pop(request.request_id, None)
        if not self._is_producer:
            self._producer_layer_topk_by_request.pop(
                request.request_id, None
            )
            return False, None
        params = self._build_producer_params(request)
        if params is None:
            return False, None
        if self._indexer_group_id is None or self._resident_group_id is None:
            raise RuntimeError(
                "DSAMooncakeConnector scheduler is missing cache groups"
            )
        if len(block_ids) <= max(
            self._indexer_group_id, self._resident_group_id
        ):
            raise RuntimeError(
                "DSAMooncakeConnector did not receive all cache-group block "
                "tables"
            )
        manifest = DSAKVIOPDManifest.from_dict(
            params[DSA_KVIO_PD_MANIFEST_KEY]
        )
        prompt_blocks = int(manifest.logical_block_count)
        indexer_blocks = list(block_ids[self._indexer_group_id])
        resident_blocks = list(block_ids[self._resident_group_id])
        if (
            len(indexer_blocks) < prompt_blocks
            or len(resident_blocks) < prompt_blocks
        ):
            raise RuntimeError(
                "DSA Mooncake P cache was released before handoff metadata "
                "was built: "
                f"required={prompt_blocks}, indexer={len(indexer_blocks)}, "
                f"resident={len(resident_blocks)}"
            )
        remote_group_block_ids = tuple(
            (
                indexer_blocks[:prompt_blocks]
                if group_id == self._indexer_group_id
                else resident_blocks[:prompt_blocks]
                if group_id == self._resident_group_id
                else []
            )
            for group_id in range(len(block_ids))
        )
        params.update(
            {
                DSA_PD_INITIAL_TRANSPORT_KEY:
                    DSA_PD_INITIAL_TRANSPORT_MOONCAKE,
                "remote_block_ids": remote_group_block_ids,
                "remote_engine_id": self._engine_id,
                "remote_request_id": request.request_id,
                "remote_host": self._side_channel_host,
                "remote_port": self._side_channel_port,
                "remote_multi_nodes_meta_mapping":
                    self._multi_nodes_meta_mapping,
                "num_prompt_blocks": prompt_blocks,
            }
        )
        self._requests_need_send[request.request_id] = time.time()
        return True, params

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        if self._worker is None:
            raise RuntimeError(
                "DSAMooncakeConnector scheduler cannot register KV caches"
            )
        self._worker.register_kv_caches(kv_caches)

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        del forward_context, kwargs
        if self._worker is None:
            raise RuntimeError(
                "DSAMooncakeConnector scheduler cannot load KV caches"
            )
        if not isinstance(
            self._connector_metadata, DSAMooncakeConnectorMetadata
        ):
            raise TypeError(
                "DSAMooncakeConnector received unexpected worker metadata"
            )
        self._worker.start_load_kv(self._connector_metadata)

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str], set[str]]:
        del finished_req_ids
        if self._worker is None:
            return set(), set()
        return self._worker.get_finished()

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        return

    def get_handshake_metadata(
        self,
    ) -> KVConnectorHandshakeMetadata | None:
        if self._worker is None:
            return None
        return self._worker.xfer_handshake_metadata

    def set_xfer_handshake_metadata(
        self,
        metadata: dict[int, KVConnectorHandshakeMetadata],
    ) -> None:
        for local_rank, rank_metadata in metadata.items():
            self._multi_nodes_meta_mapping[str(local_rank)] = {
                "host": rank_metadata.local_ip,
                "engine_id": rank_metadata.engine_id,
            }


class DSAMooncakeConnectorWorker(MooncakeConnectorWorker):
    """Raw TransferEngine worker with DSA-specific cache address planning."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        if (
            self._prefill_tp_size != self._decode_tp_size
            or self._prefill_dp_size != self._decode_dp_size
        ):
            raise ValueError(
                "DSAMooncakeConnector currently requires identical P/D "
                "TP and DP sizes"
            )
        if (
            self._prefill_pp_size != 1
            or self._decode_pp_size != 1
            or self.pcp_size != 1
            or self.dcp_size != 1
        ):
            raise ValueError(
                "DSAMooncakeConnector currently supports matching P/D "
                "topology with PP=PCP=DCP=1"
            )
        self._parallel_rank = int(
            getattr(vllm_config.parallel_config, "rank", self.tp_rank)
        )
        self._regions: list[_DSAMooncakeCacheRegion] = []
        self._regions_by_key: dict[
            tuple[int, str], _DSAMooncakeCacheRegion
        ] = {}
        self._region_indices_by_key: dict[tuple[int, str], int] = {}
        self._remote_metadata: dict[
            tuple[str, int, str], MooncakeAgentMetadata
        ] = {}

    def _make_region(
        self,
        *,
        layer_id: int,
        component: str,
        tensor: torch.Tensor,
        num_blocks: int,
    ) -> _DSAMooncakeCacheRegion:
        if not tensor.is_contiguous():
            raise ValueError(
                "DSA Mooncake requires contiguous cache tensors: "
                f"layer={layer_id}, component={component}"
            )
        total_bytes = int(tensor.numel()) * int(tensor.element_size())
        if num_blocks <= 0 or total_bytes % num_blocks:
            raise ValueError(
                "DSA Mooncake cache bytes are not divisible by physical "
                f"blocks: layer={layer_id}, component={component}"
            )
        block_bytes = total_bytes // num_blocks
        if block_bytes % self.block_size:
            raise ValueError(
                "DSA Mooncake cache block bytes are not divisible by block "
                f"size: layer={layer_id}, component={component}"
            )
        return _DSAMooncakeCacheRegion(
            layer_id=int(layer_id),
            component=component,
            tensor=tensor,
            num_blocks=int(num_blocks),
            block_bytes=block_bytes,
            token_bytes=block_bytes // self.block_size,
        )

    def _build_regions(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> list[_DSAMooncakeCacheRegion]:
        indexer_groups = [
            group
            for group in self.kv_cache_config.kv_cache_groups
            if is_dsa_indexer_spec(group.kv_cache_spec)
        ]
        resident_groups = [
            group
            for group in self.kv_cache_config.kv_cache_groups
            if is_dsa_mla_resident_spec(group.kv_cache_spec)
        ]
        if len(indexer_groups) != 1 or len(resident_groups) != 1:
            raise ValueError(
                "DSAMooncakeConnector requires one Indexer and one MLA group"
            )
        indexer_group, resident_group = (
            indexer_groups[0], resident_groups[0]
        )
        indexer_by_layer = {
            _layer_id(name): kv_caches[name]
            for name in indexer_group.layer_names
        }
        resident_by_layer = {
            _layer_id(name): kv_caches[name]
            for name in resident_group.layer_names
        }
        if set(indexer_by_layer) != set(resident_by_layer):
            raise ValueError(
                "DSA Mooncake Indexer and MLA layer sets do not match"
            )
        indexer_num_blocks = int(
            getattr(indexer_group, "dsa_num_blocks",
                    self.kv_cache_config.num_blocks)
        )
        resident_num_blocks = int(
            getattr(resident_group, "dsa_num_blocks",
                    self.kv_cache_config.num_blocks)
        )
        regions: list[_DSAMooncakeCacheRegion] = []
        for layer_id in sorted(indexer_by_layer):
            indexer_cache = indexer_by_layer[layer_id]
            resident_cache = resident_by_layer[layer_id]
            if not torch.is_tensor(indexer_cache):
                raise TypeError(
                    f"DSA Mooncake layer {layer_id} Indexer is not a tensor"
                )
            if (
                not isinstance(resident_cache, (tuple, list))
                or len(resident_cache) < 2
                or not torch.is_tensor(resident_cache[0])
                or not torch.is_tensor(resident_cache[1])
            ):
                raise TypeError(
                    f"DSA Mooncake layer {layer_id} MLA cache is invalid"
                )
            regions.extend(
                (
                    self._make_region(
                        layer_id=layer_id,
                        component=_INDEXER_COMPONENT,
                        tensor=indexer_cache,
                        num_blocks=indexer_num_blocks,
                    ),
                    self._make_region(
                        layer_id=layer_id,
                        component=_NOPE_COMPONENT,
                        tensor=resident_cache[0],
                        num_blocks=resident_num_blocks,
                    ),
                    self._make_region(
                        layer_id=layer_id,
                        component=_ROPE_COMPONENT,
                        tensor=resident_cache[1],
                        num_blocks=resident_num_blocks,
                    ),
                )
            )
        return regions

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        self.kv_caches = kv_caches
        self._regions = self._build_regions(kv_caches)
        self._regions_by_key = {
            (region.layer_id, region.component): region
            for region in self._regions
        }
        self._region_indices_by_key = {
            (region.layer_id, region.component): index
            for index, region in enumerate(self._regions)
        }
        ptrs = [int(region.tensor.data_ptr()) for region in self._regions]
        lengths = [
            int(region.tensor.numel()) * int(region.tensor.element_size())
            for region in self._regions
        ]
        global_te.register_buffer(ptrs, lengths)
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            block_size=self.block_size,
            kv_caches_base_addr=ptrs,
            num_blocks=int(self.kv_cache_config.num_blocks),
            block_lens=[
                int(region.block_bytes) for region in self._regions
            ],
            ssm_sizes=(0, 0),
            local_ip=get_ip(),
        )
        self.xfer_handshake_metadata = metadata
        if self.kv_role != "kv_producer":
            return
        ready_event = threading.Event()
        self.kv_send_thread = KVCacheSendingThread(
            self.vllm_config,
            self.tp_rank,
            self._prefill_tp_size,
            self.engine_id,
            self.side_channel_host,
            self.side_channel_port,
            metadata,
            ready_event,
            self.kv_caches,
            self.pcp_rank,
        )
        self.kv_send_thread.start()
        if not ready_event.wait(timeout=300):
            raise RuntimeError(
                "Timeout waiting for DSA Mooncake metadata server"
            )
        if not self.kv_send_thread.is_alive():
            raise RuntimeError(
                "DSA Mooncake metadata server failed to start"
            )

    def _remote_endpoint(
        self,
        request: DSAKVIOPDRequest,
    ) -> tuple[str, int, str]:
        if (
            request.remote_host is None
            or request.remote_port is None
            or request.remote_engine_id is None
        ):
            raise RuntimeError(
                "DSA Mooncake request has incomplete remote endpoint"
            )
        port_offset = int(self.handshake_port - self.side_channel_port)
        mapping = request.remote_multi_nodes_meta_mapping or {}
        rank_info = (
            mapping.get(str(port_offset))
            or mapping.get(str(self._parallel_rank))
            or {}
        )
        return (
            str(rank_info.get("host", request.remote_host)),
            int(request.remote_port) + port_offset,
            str(rank_info.get("engine_id", request.remote_engine_id)),
        )

    @staticmethod
    def _open_remote_socket(
        remote_host: str,
        remote_port: int,
    ) -> tuple[zmq.Context, zmq.Socket, zmq.Poller, str]:
        path = make_zmq_path("tcp", remote_host, remote_port)
        context = zmq.Context()  # type: ignore
        socket = make_zmq_socket(
            ctx=context,
            path=path,
            socket_type=zmq.REQ,  # type: ignore
            bind=False,
        )
        poller = zmq.Poller()  # type: ignore
        poller.register(socket, zmq.POLLIN)  # type: ignore
        return context, socket, poller, path

    def _get_remote_metadata(
        self,
        remote_host: str,
        remote_port: int,
        remote_engine_id: str,
    ) -> MooncakeAgentMetadata:
        cache_key = (remote_host, remote_port, remote_engine_id)
        cached = self._remote_metadata.get(cache_key)
        if cached is not None:
            return cached
        context, socket, poller, path = self._open_remote_socket(
            remote_host, remote_port
        )
        try:
            payload = msgspec.msgpack.Encoder().encode(
                (GET_META_MSG, "")
            )
            ensure_zmq_send(socket, payload, path)
            encoded = ensure_zmq_recv(socket, poller, path)
            metadata = msgspec.msgpack.Decoder(
                MooncakeAgentMetadata
            ).decode(encoded)
        finally:
            socket.close(linger=0)
            context.term()
        if metadata.engine_id != remote_engine_id:
            raise RuntimeError(
                "DSA Mooncake remote engine mismatch: route="
                f"{remote_engine_id}, metadata={metadata.engine_id}"
            )
        if metadata.block_size != self.block_size:
            raise RuntimeError(
                "DSA Mooncake remote block size mismatch: "
                f"{metadata.block_size} vs {self.block_size}"
            )
        local_block_lens = [
            int(region.block_bytes) for region in self._regions
        ]
        if (
            len(metadata.kv_caches_base_addr) != len(self._regions)
            or list(metadata.block_lens) != local_block_lens
        ):
            raise RuntimeError(
                "DSA Mooncake remote cache-region layout does not match D"
            )
        self._remote_metadata[cache_key] = metadata
        return metadata

    def _send_transfer_done(
        self,
        *,
        request_id: str,
        remote_host: str,
        remote_port: int,
    ) -> None:
        context, socket, poller, path = self._open_remote_socket(
            remote_host, remote_port
        )
        try:
            payload = msgspec.msgpack.Encoder().encode(
                (DONE_RECVING_MSG, request_id, {})
            )
            ensure_zmq_send(socket, payload, path)
            response = ensure_zmq_recv(socket, poller, path)
            if response != b"ACK":
                raise RuntimeError(
                    "DSA Mooncake metadata server did not acknowledge "
                    f"request {request_id}"
                )
        finally:
            socket.close(linger=0)
            context.term()

    def _build_request_transfers(
        self,
        request: DSAKVIOPDRequest,
        remote_metadata: MooncakeAgentMetadata,
    ) -> list[_DSAMooncakeTransfer]:
        manifest = request.manifest
        manifest.validate()
        remote_indexer_blocks = request.remote_indexer_block_ids
        remote_resident_blocks = request.remote_resident_block_ids
        if remote_indexer_blocks is None or remote_resident_blocks is None:
            raise RuntimeError(
                "DSA Mooncake request has no P-side block tables"
            )
        layer_topk = request.layer_topk_by_rank.get(self._parallel_rank)
        if layer_topk is None:
            raise RuntimeError(
                "DSA Mooncake request has no TopK metadata for rank "
                f"{self._parallel_rank}"
            )
        expected_layers = {
            layer_id
            for layer_id, component in self._regions_by_key
            if component == _INDEXER_COMPONENT
        }
        if {int(layer_id) for layer_id in layer_topk} != expected_layers:
            raise RuntimeError(
                "DSA Mooncake TopK layer set does not match local caches"
            )

        transfers: list[_DSAMooncakeTransfer] = []
        for layer_id in sorted(expected_layers):
            indexer_region = self._regions_by_key[
                (layer_id, _INDEXER_COMPONENT)
            ]
            indexer_region_index = self._region_indices_by_key[
                (layer_id, _INDEXER_COMPONENT)
            ]
            _append_contiguous_range_mapping(
                transfers,
                local_region=indexer_region,
                remote_base_address=remote_metadata.kv_caches_base_addr[
                    indexer_region_index
                ],
                remote_block_ids=remote_indexer_blocks,
                local_block_ids=request.indexer_block_ids,
                source_token_start=0,
                destination_slot_start=0,
                token_count=manifest.stored_token_count,
                block_size=self.block_size,
            )
            resident_token_ids = build_pd_resident_token_ids(
                topk_token_ids=layer_topk[layer_id],
                stored_token_count=manifest.stored_token_count,
                block_size=manifest.block_size,
                resident_token_count=manifest.resident_token_count,
            )
            resident_slots = list(range(len(resident_token_ids)))
            for component in (_NOPE_COMPONENT, _ROPE_COMPONENT):
                region = self._regions_by_key[(layer_id, component)]
                region_index = self._region_indices_by_key[
                    (layer_id, component)
                ]
                remote_base = remote_metadata.kv_caches_base_addr[
                    region_index
                ]
                _append_token_mapping(
                    transfers,
                    local_region=region,
                    remote_base_address=remote_base,
                    remote_block_ids=remote_resident_blocks,
                    local_block_ids=request.resident_block_ids,
                    source_token_ids=resident_token_ids,
                    destination_slots=resident_slots,
                    block_size=self.block_size,
                )
                _append_contiguous_range_mapping(
                    transfers,
                    local_region=region,
                    remote_base_address=remote_base,
                    remote_block_ids=remote_resident_blocks,
                    local_block_ids=request.resident_block_ids,
                    source_token_start=manifest.tail_token_start,
                    destination_slot_start=manifest.tail_slot_start,
                    token_count=manifest.tail_token_count,
                    block_size=self.block_size,
                )
        return transfers

    def _load_request(self, request: DSAKVIOPDRequest) -> None:
        if request.remote_request_id is None:
            raise RuntimeError(
                "DSA Mooncake request has no P-side request id"
            )
        remote_host, remote_port, remote_engine_id = (
            self._remote_endpoint(request)
        )
        remote_metadata = self._get_remote_metadata(
            remote_host, remote_port, remote_engine_id
        )
        transfers = self._build_request_transfers(
            request, remote_metadata
        )
        if not transfers:
            raise RuntimeError(
                f"DSA Mooncake built no transfers for {request.request_id}"
            )
        session_id = f"{remote_host}:{remote_metadata.te_rpc_port}"
        started = time.perf_counter()
        for start in range(0, len(transfers), _MOONCAKE_TRANSFER_BATCH_SIZE):
            batch = transfers[
                start:start + _MOONCAKE_TRANSFER_BATCH_SIZE
            ]
            result = self.engine.batch_transfer_sync_read(
                session_id,
                [item.local_address for item in batch],
                [item.remote_address for item in batch],
                [item.length for item in batch],
            )
            if result < 0:
                raise RuntimeError(
                    "DSA Mooncake transfer failed: "
                    f"request={request.request_id}, ret={result}"
                )
        self._send_transfer_done(
            request_id=request.remote_request_id,
            remote_host=remote_host,
            remote_port=remote_port,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "DSA Mooncake initial handoff completed: request=%s, "
            "transfers=%d, bytes=%d, elapsed_ms=%.2f",
            request.request_id,
            len(transfers),
            sum(item.length for item in transfers),
            elapsed_ms,
        )

    def start_load_kv(
        self,
        metadata: DSAMooncakeConnectorMetadata,
    ) -> None:
        if self.kv_role == "kv_producer":
            if self.kv_send_thread is None:
                raise RuntimeError(
                    "DSA Mooncake producer metadata server is not running"
                )
            for request_id in metadata.reqs_in_batch:
                self.kv_send_thread.task_tracker.add_req_to_process(
                    request_id
                )
            for request_id, delay_start in (
                metadata.requests_to_send.items()
            ):
                self.kv_send_thread.add_delayed_request(
                    request_id, delay_start
                )
            return
        for request in metadata.dsa_requests:
            self._load_request(request)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_thread.get_and_clear_finished_requests()
            if self.kv_role == "kv_producer"
            and self.kv_send_thread is not None
            else set()
        )
        # D loads are synchronous in start_load_kv(), so the scheduler never
        # places them in the asynchronous WAITING_FOR_REMOTE_KVS state.
        return done_sending, set()
