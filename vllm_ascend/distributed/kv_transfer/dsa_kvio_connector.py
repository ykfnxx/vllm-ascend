"""vLLM V1 connector control plane for DSA KVIO P/D separation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

from vllm_ascend.dsa_sparse.dsa_kvio_backend import KVIODSAKVBackend
from vllm_ascend.dsa_sparse.dsa_pd import (
    DSA_KVIO_CONNECTOR_NAME,
    DSA_KVIO_PD_LAYER_TOPK_KEY,
    DSA_KVIO_PD_MANIFEST_KEY,
    DSA_PD_INITIAL_TRANSPORT_KEY,
    DSA_PD_INITIAL_TRANSPORT_KVIO,
    DSAKVIOPDManifest,
    DSAKVIOPDRequest,
    build_dsa_kvio_layout_fingerprint,
    build_pd_resident_token_ids,
    get_dsa_kvio_layer_topk,
    get_dsa_kvio_pd_manifest,
    serialize_dsa_kvio_layer_topk,
)
from vllm_ascend.dsa_sparse.dsa_resident_pool import (
    DSA_LOOKUP_INDEX_CAPACITY,
    DSA_LOOKUP_QUERY_TOKENS,
    DSA_LOOKUP_RESIDENT_TOKENS,
    DSA_LOOKUP_TOTAL_SLOTS,
)
from vllm_ascend.dsa_sparse.dsa_spec_utils import (
    is_dsa_indexer_spec,
    is_dsa_mla_resident_spec,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger("vllm.dsa_sparse")


@dataclass
class DSAKVIOConnectorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for D-side cache initialization."""

    dsa_requests: list[DSAKVIOPDRequest] = field(default_factory=list)


@dataclass
class DSAKVIOConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    """P-worker TopK positions, grouped by request and global worker rank."""

    request_layer_topk_by_rank: dict[
        str, dict[int, dict[int, list[int]]]
    ] = field(default_factory=dict)

    def aggregate(
        self,
        other: KVConnectorWorkerMetadata,
    ) -> "DSAKVIOConnectorWorkerMetadata":
        if not isinstance(other, DSAKVIOConnectorWorkerMetadata):
            raise TypeError(
                "Cannot aggregate DSA KVIO metadata with "
                f"{type(other).__name__}"
            )
        for request_id, other_by_rank in (
            other.request_layer_topk_by_rank.items()
        ):
            by_rank = self.request_layer_topk_by_rank.setdefault(
                request_id, {}
            )
            for rank, other_layers in other_by_rank.items():
                layers = by_rank.setdefault(int(rank), {})
                for layer_id, other_topk in other_layers.items():
                    current = layers.get(int(layer_id))
                    normalized = [int(token_id) for token_id in other_topk]
                    if current is not None and current != normalized:
                        raise RuntimeError(
                            "DSA KVIO workers produced conflicting TopK "
                            f"metadata: request={request_id}, rank={rank}, "
                            f"layer={layer_id}"
                        )
                    layers[int(layer_id)] = normalized
        return self


class DSAKVIOConnector(KVConnectorBase_V1, SupportsHMA):
    """Use vLLM's connector lifecycle as KVIO's P/D control plane.

    KV bytes remain in KVIO.  This connector returns a compact P-built
    manifest plus per-rank layer TopK through ``kv_transfer_params`` and
    forwards D-side physical block allocations to ``DSASparseV1``.  The DSA
    worker manager performs the synchronous KVIO GET before the first D-side
    model forward.
    """

    _initial_transport = DSA_PD_INITIAL_TRANSPORT_KVIO

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        cache_config = vllm_config.cache_config
        if not getattr(cache_config, "enable_dsa_sparse_cache", False):
            raise ValueError("DSAKVIOConnector requires DSA sparse cache")
        if getattr(cache_config, "dsa_kv_backend", None) != "kvio":
            raise ValueError(
                "DSAKVIOConnector requires "
                "dsa_sparse_config['kv_backend']='kvio'")
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError(
                "DSAKVIOConnector does not support speculative decoding")
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        if bool(getattr(scheduler_config, "async_scheduling", False)):
            raise ValueError(
                "DSAKVIOConnector requires async scheduling to be disabled")
        self._block_size = int(cache_config.block_size)
        self._sparse_handoff_threshold = (
            DSA_LOOKUP_TOTAL_SLOTS + self._block_size
        )
        self._model_id = int(cache_config.dsa_kvio_model_id)
        self._layout_fingerprint = build_dsa_kvio_layout_fingerprint(
            vllm_config)
        self._engine_id = str(
            getattr(self._kv_transfer_config, "engine_id", ""))
        parallel_config = getattr(vllm_config, "parallel_config", None)
        self._rank = int(getattr(parallel_config, "rank", 0))
        self._world_size = int(getattr(parallel_config, "world_size", 1))
        if self._world_size <= 0:
            raise ValueError(
                f"{DSA_KVIO_CONNECTOR_NAME} requires world_size > 0")
        self._requests_need_load: dict[str, DSAKVIOPDRequest] = {}
        self._producer_layer_topk_by_request: dict[
            str, dict[int, dict[int, list[int]]]
        ] = {}

        if kv_cache_config is None:
            self._indexer_group_id = None
            self._resident_group_id = None
        else:
            indexer_groups = [
                group_id
                for group_id, group in enumerate(
                    kv_cache_config.kv_cache_groups)
                if is_dsa_indexer_spec(group.kv_cache_spec)
            ]
            resident_groups = [
                group_id
                for group_id, group in enumerate(
                    kv_cache_config.kv_cache_groups)
                if is_dsa_mla_resident_spec(group.kv_cache_spec)
            ]
            if len(indexer_groups) != 1 or len(resident_groups) != 1:
                raise ValueError(
                    "DSAKVIOConnector requires exactly one Indexer group and "
                    "one MLA resident group")
            self._indexer_group_id = indexer_groups[0]
            self._resident_group_id = resident_groups[0]

    @property
    def _is_producer(self) -> bool:
        return bool(self._kv_transfer_config.is_kv_producer)

    @property
    def _is_consumer(self) -> bool:
        return bool(self._kv_transfer_config.is_kv_consumer)

    def _validate_manifest(
        self,
        manifest: DSAKVIOPDManifest,
    ) -> None:
        manifest.validate()
        if manifest.block_size != self._block_size:
            raise ValueError(
                "DSA KVIO P/D block size mismatch: producer="
                f"{manifest.block_size}, consumer={self._block_size}")
        if manifest.model_id != self._model_id:
            raise ValueError(
                "DSA KVIO P/D model id mismatch: producer="
                f"{manifest.model_id}, consumer={self._model_id}")
        if manifest.layout_fingerprint != self._layout_fingerprint:
            raise ValueError(
                "DSA KVIO P/D cache layout fingerprint mismatch: producer="
                f"{manifest.layout_fingerprint}, "
                f"consumer={self._layout_fingerprint}")
        if manifest.producer_world_size != self._world_size:
            raise ValueError(
                "DSA KVIO P/D parallel world size mismatch: producer="
                f"{manifest.producer_world_size}, consumer={self._world_size}")
        if manifest.index_capacity != DSA_LOOKUP_INDEX_CAPACITY:
            raise ValueError("DSA KVIO P/D lookup index capacity mismatch")
        if manifest.resident_tokens != DSA_LOOKUP_RESIDENT_TOKENS:
            raise ValueError("DSA KVIO P/D resident token count mismatch")
        if manifest.free_slot_tokens != DSA_LOOKUP_QUERY_TOKENS:
            raise ValueError("DSA KVIO P/D free-slot token count mismatch")

    @staticmethod
    def _validate_layer_topk_ranks(
        manifest: DSAKVIOPDManifest,
        layer_topk_by_rank: dict[int, dict[int, list[int]]],
    ) -> None:
        expected_ranks = set(range(manifest.producer_world_size))
        actual_ranks = {int(rank) for rank in layer_topk_by_rank}
        if actual_ranks != expected_ranks:
            raise ValueError(
                "DSA KVIO P/D per-layer TopK rank set mismatch: "
                f"expected={sorted(expected_ranks)}, "
                f"actual={sorted(actual_ranks)}")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if not self._is_consumer:
            return 0, False
        manifest = get_dsa_kvio_pd_manifest(request.kv_transfer_params)
        if manifest is None:
            return 0, False
        transfer_params = request.kv_transfer_params
        if (
            not isinstance(transfer_params, dict)
            or not transfer_params.get("do_remote_prefill", False)
        ):
            raise ValueError(
                "DSA KVIO P/D manifest requires do_remote_prefill=true")
        initial_transport = transfer_params.get(
            DSA_PD_INITIAL_TRANSPORT_KEY,
            DSA_PD_INITIAL_TRANSPORT_KVIO,
        )
        if initial_transport != self._initial_transport:
            raise ValueError(
                f"{type(self).__name__} received initial transport "
                f"{initial_transport!r}, expected {self._initial_transport!r}"
            )
        layer_topk_by_rank = get_dsa_kvio_layer_topk(
            transfer_params
        )
        if not layer_topk_by_rank:
            raise ValueError(
                "DSA KVIO P/D handoff is missing per-layer Prefill TopK"
            )
        self._validate_manifest(manifest)
        self._validate_layer_topk_ranks(manifest, layer_topk_by_rank)
        if int(num_computed_tokens) != 0:
            raise RuntimeError(
                "DSA KVIO P/D does not support mixing a local prefix-cache "
                "hit with remote DSA state; disable prefix caching on D")
        prompt_token_ids = request.prompt_token_ids
        if not prompt_token_ids:
            raise ValueError(
                "DSA KVIO P/D handoff requires tokenized D prompt input")
        if "last_token_id" not in transfer_params:
            raise ValueError(
                "DSA KVIO P/D handoff is missing last_token_id")
        last_token_id = int(transfer_params["last_token_id"])
        if int(prompt_token_ids[-1]) != last_token_id:
            raise ValueError(
                "DSA KVIO P/D handoff token mismatch: route appended "
                f"{int(prompt_token_ids[-1])}, producer returned "
                f"{last_token_id}")
        expected_stored_tokens = max(0, len(prompt_token_ids) - 1)
        if manifest.stored_token_count != expected_stored_tokens:
            raise ValueError(
                "DSA KVIO P/D prompt handoff mismatch: manifest stores "
                f"{manifest.stored_token_count} tokens but D request expects "
                f"{expected_stored_tokens}")
        return manifest.stored_token_count, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if not self._is_consumer or int(num_external_tokens) <= 0:
            return
        manifest = get_dsa_kvio_pd_manifest(request.kv_transfer_params)
        if manifest is None:
            raise RuntimeError(
                "DSA KVIO P/D external allocation is missing its manifest")
        self._validate_manifest(manifest)
        layer_topk_by_rank = get_dsa_kvio_layer_topk(
            request.kv_transfer_params
        )
        if not layer_topk_by_rank:
            raise RuntimeError(
                "DSA KVIO P/D allocation is missing per-layer Prefill TopK"
            )
        self._validate_layer_topk_ranks(manifest, layer_topk_by_rank)
        if self._indexer_group_id is None or self._resident_group_id is None:
            raise RuntimeError(
                "DSAKVIOConnector scheduler is missing KV cache group config")
        block_ids = blocks.get_block_ids()
        self._requests_need_load[request.request_id] = DSAKVIOPDRequest(
            request_id=request.request_id,
            manifest=manifest,
            indexer_block_ids=list(block_ids[self._indexer_group_id]),
            resident_block_ids=list(block_ids[self._resident_group_id]),
            layer_topk_by_rank=layer_topk_by_rank,
        )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        metadata = DSAKVIOConnectorMetadata()
        for request_id in scheduler_output.num_scheduled_tokens:
            request = self._requests_need_load.pop(request_id, None)
            if request is not None:
                metadata.dsa_requests.append(request)
        return metadata

    def _build_producer_params(
        self,
        request: "Request",
    ) -> dict[str, Any] | None:
        if not self._is_producer:
            return None
        layer_topk_by_rank = self._producer_layer_topk_by_request.pop(
            request.request_id, None
        )
        request_transfer_params = getattr(
            request, "kv_transfer_params", None)
        if (
            not isinstance(request_transfer_params, dict)
            or not request_transfer_params.get("do_remote_decode", False)
        ):
            return None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            logger.debug(
                "DSA KVIO P/D request %s stopped with status %s; no decode "
                "handoff will be produced",
                request.request_id,
                request.status,
            )
            return None
        output_token_ids = list(
            getattr(request, "output_token_ids", ()) or ())
        if not output_token_ids:
            logger.warning(
                "DSA KVIO P/D request %s produced no handoff token",
                request.request_id,
            )
            return None
        if len(output_token_ids) != 1:
            raise RuntimeError(
                "DSA KVIO P/D producer must generate exactly one token before "
                f"handoff: request={request.request_id}, "
                f"generated={len(output_token_ids)}")
        stored_token_count = min(
            int(request.num_computed_tokens),
            int(request.num_prompt_tokens),
        )
        if stored_token_count <= 0:
            return None
        if stored_token_count < int(request.num_prompt_tokens):
            logger.warning(
                "DSA KVIO P/D request %s finished before its full prompt was "
                "stored; no handoff manifest will be produced",
                request.request_id,
            )
            return None
        if stored_token_count + 1 <= self._sparse_handoff_threshold:
            logger.debug(
                "DSA KVIO P/D request %s has %d tokens after appending the "
                "handoff token, which does not exceed the sparse threshold "
                "%d; no compact handoff will be produced",
                request.request_id,
                stored_token_count + 1,
                self._sparse_handoff_threshold,
            )
            return None
        remote_request_id = KVIODSAKVBackend.encode_request_id(
            request.request_id,
            namespace=self._engine_id,
        )
        manifest = DSAKVIOPDManifest.build(
            remote_request_id=remote_request_id,
            model_id=self._model_id,
            stored_token_count=stored_token_count,
            block_size=self._block_size,
            index_capacity=DSA_LOOKUP_INDEX_CAPACITY,
            resident_tokens=DSA_LOOKUP_RESIDENT_TOKENS,
            free_slot_tokens=DSA_LOOKUP_QUERY_TOKENS,
            generation=remote_request_id,
            producer_world_size=self._world_size,
            layout_fingerprint=self._layout_fingerprint,
        )
        if not layer_topk_by_rank:
            raise RuntimeError(
                "DSA KVIO P/D request finished without the last Prefill "
                f"token's layer TopK: request={request.request_id}"
            )
        # Validate the compact TopK seed on P.  D deterministically expands it
        # to the exact same 8192-token resident membership after allocation.
        for layers in layer_topk_by_rank.values():
            for topk_token_ids in layers.values():
                build_pd_resident_token_ids(
                    topk_token_ids=topk_token_ids,
                    stored_token_count=manifest.stored_token_count,
                    block_size=manifest.block_size,
                    resident_token_count=manifest.resident_token_count,
                )
        self._validate_layer_topk_ranks(manifest, layer_topk_by_rank)
        return {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "last_token_id": int(output_token_ids[-1]),
            DSA_PD_INITIAL_TRANSPORT_KEY: DSA_PD_INITIAL_TRANSPORT_KVIO,
            DSA_KVIO_PD_MANIFEST_KEY: manifest.to_dict(),
            DSA_KVIO_PD_LAYER_TOPK_KEY: serialize_dsa_kvio_layer_topk(
                layer_topk_by_rank
            ),
        }

    def _consume_worker_metadata(
        self,
        worker_metadata: KVConnectorWorkerMetadata | None,
    ) -> None:
        if worker_metadata is None:
            return
        if not isinstance(worker_metadata, DSAKVIOConnectorWorkerMetadata):
            raise TypeError(
                "DSAKVIOConnector received unexpected worker metadata: "
                f"{type(worker_metadata).__name__}"
            )
        for request_id, incoming_by_rank in (
            worker_metadata.request_layer_topk_by_rank.items()
        ):
            by_rank = self._producer_layer_topk_by_request.setdefault(
                request_id, {}
            )
            for rank, incoming_layers in incoming_by_rank.items():
                layers = by_rank.setdefault(int(rank), {})
                for layer_id, token_ids in incoming_layers.items():
                    layers[int(layer_id)] = list(token_ids)

    def build_connector_worker_meta(
        self,
    ) -> KVConnectorWorkerMetadata | None:
        if not self._is_producer:
            return None
        # Imported lazily to avoid coupling connector registration to worker
        # initialization order.
        from vllm_ascend.utils import get_dsa_mgr_worker

        dsa_mgr = get_dsa_mgr_worker()
        if dsa_mgr is None:
            return None
        request_layer_topk = dsa_mgr.take_pd_prefill_layer_topk()
        if not request_layer_topk:
            return None
        return DSAKVIOConnectorWorkerMetadata({
            str(request_id): {self._rank: layers}
            for request_id, layers in request_layer_topk.items()
        })

    def update_connector_output(self, connector_output) -> None:
        self._consume_worker_metadata(
            getattr(connector_output, "kv_connector_worker_meta", None)
        )

    def update_dsa_prefill_seeds_before_request_finish(
        self,
        connector_output,
    ) -> None:
        """Consume final-Prefill TopK before Scheduler frees P requests."""
        worker_metadata = getattr(
            connector_output, "kv_connector_worker_meta", None
        )
        if not isinstance(worker_metadata, DSAKVIOConnectorWorkerMetadata):
            return
        self._consume_worker_metadata(worker_metadata)
        # Scheduler.update_from_output normally consumes this after request
        # completion.  Clear only this DSA payload to prevent a stale seed from
        # being reinserted after request_finished pops it.
        connector_output.kv_connector_worker_meta = None

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        _ = block_ids
        self._requests_need_load.pop(request.request_id, None)
        if not self._is_producer:
            self._producer_layer_topk_by_request.pop(request.request_id, None)
        return False, self._build_producer_params(request)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        _ = block_ids
        self._requests_need_load.pop(request.request_id, None)
        if not self._is_producer:
            self._producer_layer_topk_by_request.pop(request.request_id, None)
        return False, self._build_producer_params(request)

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        # DSASparseV1 owns registration so one rdma_kv_ops aiv_init call covers
        # Indexer and MLA regions in a stable order on both P and D.
        return

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        # The DSA manager consumes DSAKVIOConnectorMetadata after attention
        # block tables are materialized and performs the synchronous GET.
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # DSA attention_finished writes Indexer + MLA through DSAKVBackend.
        return

    def wait_for_save(self) -> None:
        # KVIODSAKVBackend PUT is synchronous (aiv_put_batch + aiv_wait).
        return
