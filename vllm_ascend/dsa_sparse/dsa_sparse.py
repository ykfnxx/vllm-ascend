"""DSA 稀疏卸载的主生命周期管理器。

本文件保留 DSA 稀疏卸载的运行时主流程：scheduler/worker 两侧请求状态、
slot 估算、model-forward 元数据汇聚、attention_begin/after_indexer/
attention_finished hook，以及和图模式 capture/replay 的生命周期衔接。

不再把所有辅助逻辑都堆在这里：
- dsa_attention_layout.py 负责从 attention metadata 中抽取 forward 共享布局。
- dsa_batch_tensor_utils.py 负责把 request rows 物化成张量/block-table。
- dsa_forward_batch.py 负责一轮 forward 的 batch 数据结构和构造。
- dsa_graph_buffers.py 负责当前 V1 图模式下需要稳定地址的持久 buffer。

后续继续拆分时，应优先围绕元数据职责边界做小步整理；DSASparseBase 和
DSASparseV1 仍作为算法核心，保留 scheduler/worker 侧主要 hook 的入口。
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from vllm_ascend.dsa_sparse.dsa_forward_batch import (
    DSAForwardLayerBatch, DSAForwardSparseDecodeBatch, DSALayerRuntimeBatch,
    DSALayerSparseDecodeBatch, DSAModelForwardMeta,
    _build_forward_batches_from_dsa_meta)
from vllm_ascend.dsa_sparse.dsa_graph_buffers import DSAGraphBuffersMixin
from vllm_ascend.dsa_sparse.dsa_kv_backend import DSAKVBackend
from vllm_ascend.dsa_sparse.dsa_pd import (
    DSA_PD_INITIAL_TRANSPORT_KVIO,
    DSA_PD_INITIAL_TRANSPORT_MOONCAKE,
    DSAKVIOPDRequest,
    build_pd_resident_token_ids,
    get_dsa_kvio_pd_manifest,
)
from vllm_ascend.dsa_sparse.dsa_attention_layout import (
    materialize_query_position_metadata, slice_position_row,
    resolve_full_block_table_tensor, select_forward_shared_metadata)
from vllm_ascend.dsa_sparse.dsa_resident_pool import (
    DSA_LOOKUP_INDEX_CAPACITY, DSA_LOOKUP_QUERY_TOKENS,
    DSA_LOOKUP_RESIDENT_TOKENS, DSA_LOOKUP_TOTAL_SLOTS,
    DSAResidentLayerResourceView, DSAResidentTokenPool)

from vllm.v1.core.sched.output import SchedulerOutput

from vllm.config import VllmConfig

from vllm.logger import init_logger
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm_ascend.dsa_sparse.dsa_layer_cache_zones import (
    DSALayerCacheRegistry, LayerCacheZones, resolve_layer_cache_zones)
from vllm_ascend.dsa_sparse.dsa_req_meta import ReqType
from vllm_ascend.dsa_sparse.dsa_types import (
    DSADecodeRowMode, DSASparseRole, INVALID_SLOT, ReqStage)
from vllm_ascend.dsa_sparse.dsa_spec_utils import (
    is_dsa_indexer_spec,
    is_dsa_mla_resident_spec,
)
from vllm.forward_context import ForwardContext

from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger("vllm.dsa_sparse")


def _query_position_row_is_empty(row) -> bool:
    if row is None:
        return True
    if torch.is_tensor(row):
        return int(row.numel()) == 0
    return len(row) == 0


def _reset_indexer_score_controls(attn_metadata) -> None:
    attn_metadata.dsa_score_topk_k = None
    attn_metadata.dsa_indexer_seq_lens = None
    attn_metadata.dsa_selection_topk_indices = None
    attn_metadata.dsa_full_batch_selection_topk_indices = None


class DSASparseBase:
    """DSA sparse offload algorithm base shared by scheduler and worker.

    DSASparseBase owns the common algorithm configuration and invariants:
    block size, fixed lookup budget, sparse enable threshold, and the
    per-forward batch placeholders consumed by eager and graph paths.

    Keep this base class focused on the algorithm contract shared by future
    DSA sparse variants.  Execution-mode helpers, such as FULL graph stable
    buffers, should be added by concrete implementations instead of being a
    hard dependency of every DSA algorithm base.
    """

    def __init__(self, vllm_config: VllmConfig, role):
        self._vllm_config = vllm_config
        self._role = role
        self._vllm_blk_size = vllm_config.cache_config.block_size
        self._hbm_sparse_budget_tokens = int(
            vllm_config.cache_config.dsa_hbm_sparse_budget)
        self._hbm_resident_tokens = int(
            vllm_config.cache_config.dsa_hbm_resident_tokens)
        if self._is_sparse_cache_enabled():
            if self._hbm_sparse_budget_tokens != DSA_LOOKUP_QUERY_TOKENS:
                raise ValueError(
                    "DSA lookup operator requires hbm_sparse_budget="
                    f"{DSA_LOOKUP_QUERY_TOKENS}, got "
                    f"{self._hbm_sparse_budget_tokens}")
            if self._hbm_resident_tokens != DSA_LOOKUP_RESIDENT_TOKENS:
                raise ValueError(
                    "DSA lookup operator requires hbm_resident_tokens="
                    f"{DSA_LOOKUP_RESIDENT_TOKENS}, got "
                    f"{self._hbm_resident_tokens}")
            if (DSA_LOOKUP_RESIDENT_TOKENS % self._vllm_blk_size != 0
                    or DSA_LOOKUP_QUERY_TOKENS % self._vllm_blk_size != 0):
                raise ValueError(
                    "DSA lookup resident and query counts must be divisible "
                    "by block_size: "
                    f"resident={DSA_LOOKUP_RESIDENT_TOKENS}, "
                    f"query={DSA_LOOKUP_QUERY_TOKENS}, "
                    f"block_size={self._vllm_blk_size}")
        # One full TopK of free slots guarantees that lookup can allocate an
        # all-miss query before maintenance reclaims the same number of old
        # non-protected entries.
        self._lookup_free_slot_tokens = self._hbm_sparse_budget_tokens
        self._lookup_total_slot_tokens = (
            DSA_LOOKUP_TOTAL_SLOTS
            if self._is_sparse_cache_enabled()
            else self._hbm_resident_tokens + self._lookup_free_slot_tokens)
        # Sparse decode starts only after the original sequence exceeds the
        # physical lookup address space plus its independent dense tail block.
        self._enable_dsa_prompt_len = (
            self._lookup_total_slot_tokens + self._vllm_blk_size)
        self.dsa_meta = None
        self.forward_sparse_decode_batch = DSAForwardSparseDecodeBatch.empty()
        self.forward_layer_batch = DSAForwardLayerBatch.empty()
        self._forward_sparse_decode_attention_indices_tensor = None
        # FULL graph captures tensor addresses, not Python objects. DSA graph
        # paths therefore swap the normal per-forward sparse batch with
        # phase-specific persistent buffers and only refresh their contents
        # before replay.
        self._graph_row_mode_decode_batches: dict[
            tuple[str, int], DSAForwardSparseDecodeBatch] = {}
        self._graph_layer_id_tensors: dict[str, torch.Tensor] = {}
        self._sparse_forward_logged = False

    def _is_sparse_cache_enabled(self) -> bool:
        return bool(self._vllm_config.cache_config.enable_dsa_sparse_cache)

    def _get_fixed_sparse_budget_tokens(self, candidate_tokens: int) -> int:
        configured_budget = int(self._hbm_sparse_budget_tokens)
        if configured_budget <= 0 or candidate_tokens < configured_budget:
            return 0
        return configured_budget


class DSASparseV1(DSAGraphBuffersMixin, DSASparseBase):
    """Current lookup-resident based DSA sparse offload implementation.

    DSASparseV1 keeps the scheduler/worker hooks in the algorithm class and
    retains the existing graph-buffer interface. Lookup configuration rejects
    graph mode until the dynamic lookup/materialize/maintain path has
    capture-safe operators.
    """

    def __init__(self,
                 vllm_config,
                 role,
                 kv_backend: DSAKVBackend | None = None,
                 ops_backend: Any | None = None,
                 resident_device: torch.device | str | None = None):
        super().__init__(vllm_config, role)

        if self._role == DSASparseRole.SCHEDULER:
            return

        if kv_backend is None:
            raise RuntimeError("DSA sparse worker requires a KV backend")
        self.kv_backend = kv_backend
        if ops_backend is None:
            raise RuntimeError(
                "DSA sparse worker requires an Ascend lookup backend")
        self.ops_backend = ops_backend

        self.total_num_hidden_layers = (
            vllm_config.model_config.get_total_num_hidden_layers()
        )
        max_model_len = int(vllm_config.model_config.max_model_len)
        if max_model_len > DSA_LOOKUP_INDEX_CAPACITY:
            raise ValueError(
                "DSA lookup operator supports max_model_len up to "
                f"{DSA_LOOKUP_INDEX_CAPACITY}, got {max_model_len}")
        self.resident_token_pool = DSAResidentTokenPool(
            max_reqs=int(vllm_config.cache_config.dsa_max_active_reqs),
            num_layers=self.total_num_hidden_layers,
            index_capacity=DSA_LOOKUP_INDEX_CAPACITY,
            resident_tokens=self._hbm_resident_tokens,
            free_slot_tokens=self._lookup_free_slot_tokens,
            device=resident_device,
        )

        self.layer_cache_registry = DSALayerCacheRegistry(
            num_layers=self.total_num_hidden_layers)
        self.full_dump_done_by_pool = torch.zeros(
            (
                int(vllm_config.cache_config.dsa_max_active_reqs),
                self.total_num_hidden_layers,
            ),
            dtype=torch.bool,
            device="cpu",
        )
        self._lookup_maintain_seed = 0
        self._pd_initialized_requests: set[ReqType] = set()
        parallel_config = getattr(vllm_config, "parallel_config", None)
        self._parallel_rank = int(getattr(parallel_config, "rank", 0))
        # Every DSA request captures the final Prefill query's per-layer TopK.
        # A local Decode consumes it directly when seeding resident slots; a
        # P/D producer drains the same compact seed through its connector.
        self._prefill_layer_topk: dict[
            ReqType, dict[int, list[int]]
        ] = {}


    def _get_full_attention_group_id(self, kv_cache_config) -> int:
        # DSA residency/load applies to the MLA/full cache group, never the
        # selector-only indexer group.
        full_group_ids = [
            i for i, group in enumerate(kv_cache_config.kv_cache_groups)
            if is_dsa_mla_resident_spec(group.kv_cache_spec)
        ]
        if not full_group_ids:
            raise RuntimeError(
                "DSA requires an MLA/full resident KVSpec group for "
                "full-cache residency")
        return full_group_ids[0]

    def _get_indexer_group_id(self, kv_cache_config) -> int:
        # The indexer cache is a separate dense KV group so its block table and
        # budget do not collapse back into MLA/full-cache sparse semantics.
        indexer_group_ids = [
            i for i, group in enumerate(kv_cache_config.kv_cache_groups)
            if is_dsa_indexer_spec(group.kv_cache_spec)
        ]
        if not indexer_group_ids:
            raise RuntimeError("DSA requires an IndexerKVSpec group for indexer residency")
        return indexer_group_ids[0]

    def _get_group_num_free_blocks(self, block_pool, group_id: int) -> int:
        return block_pool.block_pools[group_id].get_num_free_blocks()

    def get_full_attention_group_id(self, kv_cache_config) -> int:
        return self._get_full_attention_group_id(kv_cache_config)

    def should_release_full_cache_after_prefill(self, request) -> bool:
        if (
            get_dsa_kvio_pd_manifest(request.kv_transfer_params) is not None
            and ReqStage.coerce(request.dsa_req_stage).is_sparse_decode
        ):
            # The D-side request was allocated directly in compact resident
            # layout. Releasing "prefill full cache" here would incorrectly
            # free its initialized lookup blocks.
            return False
        if request.num_prompt_tokens <= self._enable_dsa_prompt_len:
            return False
        # Once sparse cache is enabled, completed prefill MLA full blocks must
        # enter the DRAM-resident path before sparse decode can begin.
        return request.num_computed_tokens >= request.num_prompt_tokens

    def release_prefill_full_cache_except_tail(
        self,
        kv_cache_manager: KVCacheManager,
        request: Request,
    ) -> bool:
        """Release dense prefill full-cache blocks while keeping an unfull tail.

        DSA sparse decode reuses the prefill tail block as the final block in
        the full/MLA block table. Releasing the entire full-cache group at the
        prefill/decode boundary would also free that tail, forcing the first
        decode step to allocate an empty replacement block and losing the tail
        KV data.
        """
        full_group_id = self._get_full_attention_group_id(
            kv_cache_manager.kv_cache_config)
        full_manager = kv_cache_manager.coordinator.single_type_managers[
            full_group_id]
        request_id = request.request_id
        req_blocks = full_manager.req_to_blocks.get(request_id)
        if not req_blocks:
            return False

        preserve_tail_block = request.num_prompt_tokens % self._vllm_blk_size != 0
        preserved_tail_block = self._release_full_blocks_except_tail(
            full_manager, request_id, preserve_tail_block)
        self._append_preserved_tail_block(
            full_manager, request_id, preserved_tail_block)
        return True

    @staticmethod
    def _select_forward_shared_attn_metadata(attn_metadata):
        """Return layer-invariant attention metadata for this model forward.

        Ascend SFA may pass split metadata for indexer and full/MLA cache
        groups. DSA's forward-level query layout must come from full/MLA
        metadata when available because it owns the resident sparse plane.
        """
        return select_forward_shared_metadata(attn_metadata)

    def build_dsa_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, CachedRequestState],
        input_batch: InputBatch,
        attn_metadata,
        kv_cache_config,
        force_decode_row_mode_score_topk: int = 0,
    ):
        """Build DSA request metadata once for the current model forward.

        This is the boundary between vLLM's scheduler/input batch view and the
        DSA sparse-cache runtime. It only performs layer-invariant assembly:
        - acquire / bind resident-pool rows for requests in this forward;
        - snapshot each request's full/indexer block tables and query ranges;
        - build tensorized forward batches consumed by per-layer hooks.

        It intentionally does not compute indexer scores, choose replacement
        tokens, move KV cache data, or update layer resident mappings. Those
        actions happen later in the attention hooks and backend DSA ops.
        """
        self.dsa_meta = DSAModelForwardMeta()
        self.dsa_meta.full_block_table_tensor = (
            resolve_full_block_table_tensor(attn_metadata))

        attn_metadata = self._select_forward_shared_attn_metadata(
            attn_metadata)

        full_attention_group_id = self._get_full_attention_group_id(kv_cache_config)
        indexer_group_id = self._get_indexer_group_id(kv_cache_config)

        query_position_metadata = materialize_query_position_metadata(
            attn_metadata)
        cum_query_lens = query_position_metadata["cum_query_lens"]
        indexer_positions = query_position_metadata["indexer_positions"]
        resident_positions = query_position_metadata["resident_positions"]
        sparse_budgets = scheduler_output.req_dsa_sparse_budget_tokens
        req_stages = scheduler_output.req_dsa_stage
        connector_metadata = getattr(
            scheduler_output, "kv_connector_metadata", None)
        pd_requests = {
            request.request_id: request
            for request in getattr(connector_metadata, "dsa_requests", ())
            if isinstance(request, DSAKVIOPDRequest)
        }

        for (req_id, num_scheduled_tokens) in scheduler_output.num_scheduled_tokens.items():
            req_state = requests[req_id]
            context_full_blk_hashes = list(
                req_state.context_full_blk_hashes)
            expected_full_blocks = req_state.num_tokens // self._vllm_blk_size
            if len(context_full_blk_hashes) < expected_full_blocks:
                raise RuntimeError(
                    "DSA full-block hash metadata is incomplete for request "
                    f"{req_id}: expected_at_least={expected_full_blocks}, "
                    f"actual={len(context_full_blk_hashes)}, "
                    f"num_tokens={req_state.num_tokens}, "
                    f"block_size={self._vllm_blk_size}. The worker must receive "
                    "Request.block_hashes from the scheduler at vLLM block "
                    "granularity.")

            req_index = input_batch.req_id_to_index[req_id]
            req_block_ids = req_state.block_ids
            if full_attention_group_id >= len(req_block_ids):
                raise RuntimeError(
                    f"DSA build_dsa_meta missing full group block ids for req {req_id}: "
                    f"group={full_attention_group_id}, total_groups={len(req_block_ids)}")
            if indexer_group_id >= len(req_block_ids):
                raise RuntimeError(
                    f"DSA build_dsa_meta missing indexer group block ids for req {req_id}: "
                    f"group={indexer_group_id}, total_groups={len(req_block_ids)}")
            query_end_loc = cum_query_lens[req_index]
            query_start_loc = 0 if req_index == 0 else cum_query_lens[req_index - 1]
            query_len = query_end_loc - query_start_loc
            resident_valid_seq_len = (
                scheduler_output.req_dsa_resident_valid_seq_len[req_id])
            num_output_tokens = len(req_state.output_token_ids)
            sparse_budget_tokens = int(sparse_budgets[req_id])
            req_stage = ReqStage(req_stages[req_id])
            pd_request = pd_requests.get(req_id)
            sparse_decode_enabled = (
                req_stage.is_sparse_decode
                and sparse_budget_tokens > 0
                and (
                    num_output_tokens > 0
                    or pd_request is not None
                    or req_id in self._pd_initialized_requests
                )
                and int(resident_valid_seq_len) >= 0
            )
            query_positions_needed = (
                num_output_tokens > 0 or pd_request is not None)
            dense_query_positions = []
            resident_query_positions = []
            if query_positions_needed:
                dense_query_positions = slice_position_row(
                    indexer_positions, query_start_loc, query_len)
                resident_query_positions = slice_position_row(
                    resident_positions, query_start_loc, query_len)
            if _query_position_row_is_empty(resident_query_positions):
                resident_query_positions = dense_query_positions
            resident_pool_idx = self.resident_token_pool.acquire(req_id)
            if pd_request is not None:
                try:
                    self._initialize_pd_request(
                        request_id=req_id,
                        resident_pool_idx=resident_pool_idx,
                        pd_request=pd_request,
                    )
                except Exception:
                    # A failed initial materialization must not leave a
                    # request row bound to partially initialized P/D data.
                    self._rollback_pd_request_initialization(
                        request_id=req_id,
                        resident_pool_idx=resident_pool_idx,
                    )
                    raise

            # Query positions are recorded in both dense/indexer and resident
            # spaces because sparse SFA consumes resident positions after the
            # replacement plan is committed.
            self.dsa_meta.add_request_meta(request_id=req_id,
                                       index_in_batch=req_index,
                                       num_prompt_tokens=req_state.num_prompt_tokens,
                                       num_output_tokens=num_output_tokens,
                                       num_scheduled_tokens=num_scheduled_tokens,
                                       num_computed_tokens=req_state.num_computed_tokens,
                                       # Prefill/chunked prefill does not have
                                       # a sparse resident window yet.
                                       resident_valid_seq_len=resident_valid_seq_len,
                                       vllm_budget_block_ids=req_block_ids[full_attention_group_id],
                                       indexer_block_ids=req_block_ids[indexer_group_id],
                                       block_size=self._vllm_blk_size,
                                       query_start_loc=query_start_loc,
                                       query_len=query_len,
                                       req_context_full_blk_hashes=context_full_blk_hashes,
                                       dense_query_positions=dense_query_positions,
                                       resident_query_positions=resident_query_positions,
                                       stage=req_stage,
                                       dsa_sparse_enabled=sparse_decode_enabled,
                                       dsa_sparse_budget_tokens=sparse_budget_tokens,
                                       resident_pool_idx=resident_pool_idx,
                                       pd_remote_loaded=(
                                           req_id
                                           in self._pd_initialized_requests),
                                       )
        (
            self.forward_sparse_decode_batch,
            self.forward_layer_batch,
        ) = _build_forward_batches_from_dsa_meta(
            self.dsa_meta,
            tensor_device=self.resident_token_pool.device,
            force_decode_row_mode_score_topk=(
                force_decode_row_mode_score_topk),
        )
        self._forward_sparse_decode_attention_indices_tensor = None
        self._lookup_maintain_seed = (
            self._lookup_maintain_seed + 1) & 0x7FFFFFFF
        sparse_rows_tensor = (
            self.forward_sparse_decode_batch.sparse_local_row_indices_tensor
        )
        num_sparse_rows = int(sparse_rows_tensor.numel())
        if not self._sparse_forward_logged and num_sparse_rows > 0:
            logger.info(
                "DSA sparse worker forward mode active: requests=%s, "
                "sparse_rows=%d, score_topk=%d",
                self.forward_sparse_decode_batch.request_ids,
                num_sparse_rows,
                self.forward_sparse_decode_batch.score_topk_k,
            )
            self._sparse_forward_logged = True

    """
    EngineCore Scheduler侧逻辑
    """
    def request_begin(self, request_id, prompt_token_ids):
        token_id_count = (
            len(prompt_token_ids) if prompt_token_ids is not None else 0)
        logger.debug(
            "========== DSA TOKENIZED PROMPT =========="
            " req_id=%s prompt_token_id_len=%s block_size=%s "
            "sparse_threshold=%s",
            request_id,
            token_id_count,
            self._vllm_blk_size,
            self._enable_dsa_prompt_len,
        )

    def request_finished_in_scheduler(self, request_id):
        pass

    """
    Worker侧逻辑
    """
    def capture_prefill_last_token_topk(
        self,
        layer_name: str,
        topk_indices: torch.Tensor,
    ) -> None:
        """Capture every final-Prefill row's per-layer TopK.

        Keep valid prompt positions in score order. Tail exclusion is deferred
        until resident initialization because a short prompt can enter sparse
        decode only after later dense-decode tokens extend its history.
        """
        if self.dsa_meta is None:
            return
        layer_id = int(layer_name.split(".")[2])
        for req_meta in self.dsa_meta.requests:
            if (
                not req_meta.is_last_prefill_chunk
                or req_meta.query_len <= 0
                or bool(getattr(req_meta, "pd_remote_loaded", False))
            ):
                continue
            row_index = int(req_meta.query_start_loc + req_meta.query_len - 1)
            if row_index >= int(topk_indices.shape[0]):
                raise RuntimeError(
                    "DSA cannot locate the last Prefill token's TopK "
                    f"row: request={req_meta.request_id}, row={row_index}, "
                    f"topk_rows={int(topk_indices.shape[0])}"
                )
            # This is one intentional device-to-host synchronization per layer
            # at the final-Prefill boundary. Decode lookup remains tensorized;
            # only the compact 2048-position seed is retained across forwards.
            row_topk = (
                topk_indices[row_index]
                .detach()
                .reshape(-1)
                .to(device="cpu", dtype=torch.int64)
                .tolist()
            )
            prompt_token_count = int(req_meta.num_prompt_tokens)
            seen: set[int] = set()
            filtered_topk: list[int] = []
            for raw_token_id in row_topk:
                token_id = int(raw_token_id)
                if (
                    token_id < 0
                    or token_id >= prompt_token_count
                    or token_id in seen
                ):
                    continue
                filtered_topk.append(token_id)
                seen.add(token_id)
            self._prefill_layer_topk.setdefault(
                req_meta.request_id, {}
            )[layer_id] = filtered_topk

    def take_pd_prefill_layer_topk(
        self,
    ) -> dict[ReqType, dict[int, list[int]]]:
        """Drain P-worker TopK metadata for connector worker output."""
        captured = self._prefill_layer_topk
        self._prefill_layer_topk = {}
        return captured

    def _build_local_resident_initial_token_ids(
        self,
        *,
        layer_id: int,
        forward_batch: DSAForwardSparseDecodeBatch,
    ) -> tuple[torch.Tensor | None, list[ReqType]]:
        """Build TopK-first resident seeds for local non-P/D initialization."""
        if not forward_batch.has_lookup_init_rows:
            return None, []
        req_meta_by_id = {
            req_meta.request_id: req_meta
            for req_meta in self.dsa_meta.requests
        }
        default_tokens = list(range(self._hbm_resident_tokens))
        token_rows: list[list[int]] = []
        initialized_request_ids: list[ReqType] = []
        for request_id in forward_batch.request_ids:
            req_meta = req_meta_by_id.get(request_id)
            should_initialize = (
                req_meta is not None
                and req_meta.forward_plan.is_sparse_decode
                and req_meta.stage.is_enter_sparse_decode
                and not req_meta.pd_remote_loaded
            )
            if not should_initialize:
                token_rows.append(default_tokens)
                continue
            layer_topk = self._prefill_layer_topk.get(
                request_id, {}
            ).get(int(layer_id))
            if layer_topk is None:
                raise RuntimeError(
                    "DSA local resident initialization is missing the final "
                    "Prefill TopK: "
                    f"request={request_id}, layer={int(layer_id)}"
                )
            stored_token_count = (
                int(req_meta.forward_plan.dense_tail_start)
                + int(req_meta.forward_plan.tail_valid_token_count)
            )
            token_rows.append(build_pd_resident_token_ids(
                topk_token_ids=layer_topk,
                stored_token_count=stored_token_count,
                block_size=int(req_meta.block_size),
                resident_token_count=self._hbm_resident_tokens,
            ))
            initialized_request_ids.append(request_id)
        return (
            torch.tensor(
                token_rows,
                dtype=torch.int32,
                device=forward_batch.batch_hbm_block_table.device,
            ),
            initialized_request_ids,
        )

    def _consume_local_prefill_layer_topk(
        self,
        *,
        layer_id: int,
        request_ids: list[ReqType],
    ) -> None:
        """Release one layer's Prefill seed after successful local init."""
        for request_id in request_ids:
            layer_topk = self._prefill_layer_topk.get(request_id)
            if layer_topk is None:
                continue
            layer_topk.pop(int(layer_id), None)
            if not layer_topk:
                self._prefill_layer_topk.pop(request_id, None)

    def _mark_full_dump_done(
        self,
        resident_pool_indices: torch.Tensor,
        layer_id: int,
    ) -> None:
        if int(resident_pool_indices.numel()) == 0:
            return
        layer_id = int(layer_id)
        pool_indices = resident_pool_indices.to(
            device=self.full_dump_done_by_pool.device, dtype=torch.long)
        valid_mask = (
            (pool_indices >= 0)
            & (pool_indices < int(self.full_dump_done_by_pool.shape[0]))
        )
        if bool(valid_mask.any().item()):
            self.full_dump_done_by_pool[
                pool_indices[valid_mask], layer_id] = True

    def _clear_full_dump_done(self, request_id: ReqType) -> None:
        resident_pool_idx = self.resident_token_pool.get_index(request_id)
        if resident_pool_idx is None:
            return
        self.full_dump_done_by_pool[int(resident_pool_idx)].fill_(False)

    def get_layer_resident_resource_view_by_index(
        self,
        layer_id: int,
        *,
        pool_indices,
    ) -> DSAResidentLayerResourceView:
        return self.resident_token_pool.get_layer_resource_view_by_index(
            pool_indices,
            int(layer_id),
        )

    def register_kv_cache_tensors(self, kv_cache_config,
                                  kv_caches: dict[str, object]) -> None:
        """Register every local Indexer/MLA region before model execution."""
        full_group_id = self._get_full_attention_group_id(kv_cache_config)
        indexer_group_id = self._get_indexer_group_id(kv_cache_config)
        full_group = kv_cache_config.kv_cache_groups[full_group_id]
        indexer_group = kv_cache_config.kv_cache_groups[indexer_group_id]
        indexer_caches_by_layer: dict[int, torch.Tensor] = {}
        for indexer_layer_name in indexer_group.layer_names:
            indexer_cache = kv_caches[indexer_layer_name]
            if not torch.is_tensor(indexer_cache):
                raise RuntimeError(
                    "DSA requires one Indexer cache tensor for "
                    f"{indexer_layer_name}")
            indexer_layer_id = int(indexer_layer_name.split(".")[2])
            indexer_caches_by_layer[indexer_layer_id] = indexer_cache
        for layer_name in full_group.layer_names:
            layer_cache = kv_caches[layer_name]
            if not isinstance(layer_cache, (tuple, list)) or len(
                    layer_cache) < 2:
                raise RuntimeError(
                    f"DSA requires MLA nope/rope cache tensors for {layer_name}")
            layer_id = int(layer_name.split(".")[2])
            indexer_cache = indexer_caches_by_layer.get(layer_id)
            if indexer_cache is None:
                raise RuntimeError(
                    "DSA could not match an Indexer cache to MLA layer "
                    f"{layer_name}")
            self.kv_backend.register_layer_cache(
                layer_id=layer_id,
                block_size=int(self._vllm_blk_size),
                nopek_cache=layer_cache[0],
                ropek_cache=layer_cache[1],
                indexer_cache=indexer_cache,
            )
        self.kv_backend.finalize_cache_registration()

    def _initialize_pd_request(
        self,
        *,
        request_id: ReqType,
        resident_pool_idx: int,
        pd_request: DSAKVIOPDRequest,
    ) -> None:
        """Finalize a D-side compact layout materialized by KVIO/Mooncake."""
        if request_id in self._pd_initialized_requests:
            return
        manifest = pd_request.manifest
        manifest.validate()
        if manifest.block_size != int(self._vllm_blk_size):
            raise ValueError(
                "DSA KVIO P/D worker block size mismatch: producer="
                f"{manifest.block_size}, consumer={self._vllm_blk_size}")
        if manifest.index_capacity != int(
                self.resident_token_pool.index_capacity):
            raise ValueError(
                "DSA KVIO P/D worker lookup index capacity mismatch")
        if manifest.resident_tokens != int(
                self.resident_token_pool.resident_tokens):
            raise ValueError(
                "DSA KVIO P/D worker resident token count mismatch")
        if manifest.free_slot_tokens != int(
                self.resident_token_pool.free_slot_tokens):
            raise ValueError(
                "DSA KVIO P/D worker free-slot token count mismatch")

        expected_indexer_blocks = (
            manifest.stored_token_count + self._vllm_blk_size - 1
        ) // self._vllm_blk_size
        if len(pd_request.indexer_block_ids) < expected_indexer_blocks:
            raise RuntimeError(
                "DSA KVIO P/D Indexer block table is too short: "
                f"expected={expected_indexer_blocks}, "
                f"actual={len(pd_request.indexer_block_ids)}")
        required_resident_slots = (
            manifest.tail_slot_start + manifest.tail_token_count)
        expected_resident_blocks = (
            required_resident_slots + self._vllm_blk_size - 1
        ) // self._vllm_blk_size
        if len(pd_request.resident_block_ids) < expected_resident_blocks:
            raise RuntimeError(
                "DSA KVIO P/D resident block table is too short: "
                f"expected={expected_resident_blocks}, "
                f"actual={len(pd_request.resident_block_ids)}")

        layer_topk_token_ids = pd_request.layer_topk_by_rank.get(
            self._parallel_rank
        )
        if layer_topk_token_ids is None:
            raise RuntimeError(
                "DSA KVIO P/D metadata has no layer TopK for D worker rank "
                f"{self._parallel_rank}"
            )
        layer_resident_token_ids = {
            int(layer_id): build_pd_resident_token_ids(
                topk_token_ids=topk_token_ids,
                stored_token_count=manifest.stored_token_count,
                block_size=manifest.block_size,
                resident_token_count=manifest.resident_token_count,
            )
            for layer_id, topk_token_ids in layer_topk_token_ids.items()
        }

        self.kv_backend.bind_request(
            request_id=request_id,
            request_pool_idx=resident_pool_idx,
            remote_request_id=manifest.remote_request_id,
        )
        if pd_request.initial_transport == DSA_PD_INITIAL_TRANSPORT_KVIO:
            self.kv_backend.load_pd_request(
                request_pool_idx=resident_pool_idx,
                stored_token_count=manifest.stored_token_count,
                layer_resident_token_ids=layer_resident_token_ids,
                tail_token_start=manifest.tail_token_start,
                tail_token_count=manifest.tail_token_count,
                tail_slot_start=manifest.tail_slot_start,
                indexer_block_ids=pd_request.indexer_block_ids,
                resident_block_ids=pd_request.resident_block_ids,
            )
        elif (
            pd_request.initial_transport
            != DSA_PD_INITIAL_TRANSPORT_MOONCAKE
        ):
            raise ValueError(
                "Unsupported DSA P/D initial transport: "
                f"{pd_request.initial_transport!r}")
        self.resident_token_pool.initialize_request_layer_mappings(
            request_id=request_id,
            layer_token_ids=layer_resident_token_ids,
        )
        self.full_dump_done_by_pool[int(resident_pool_idx)].fill_(True)
        self._pd_initialized_requests.add(request_id)
        logger.info(
            "DSA P/D request initialized: req_id=%s, remote_req_id=%d, "
            "transport=%s, stored_tokens=%d, indexer_blocks=%d, "
            "resident_blocks=%d",
            request_id,
            manifest.remote_request_id,
            pd_request.initial_transport,
            manifest.stored_token_count,
            len(pd_request.indexer_block_ids),
            len(pd_request.resident_block_ids),
        )

    def _rollback_pd_request_initialization(
        self,
        *,
        request_id: ReqType,
        resident_pool_idx: int,
    ) -> None:
        """Release worker-local state after failed P/D materialization."""
        self._clear_full_dump_done(request_id)
        self.kv_backend.release_request(
            request_id=request_id,
            request_pool_idx=resident_pool_idx,
        )
        self.resident_token_pool.release(request_id)
        self._pd_initialized_requests.discard(request_id)

    def _build_layer_runtime_batch(
        self,
        layer_name: str,
        cache_zones: LayerCacheZones | None = None,
    ) -> DSALayerRuntimeBatch:
        layer_id = int(layer_name.split(".")[2])
        if cache_zones is None:
            cache_zones = self.layer_cache_registry.get(layer_id)
        return self.forward_layer_batch.layer_runtime_batch(
            layer_id=layer_id,
            cache_zones=cache_zones,
        )

    def _ensure_layer_begin_sparse_decode_dump_ready(
        self,
        begin_batch: DSALayerRuntimeBatch,
    ) -> None:
        """Guard sparse decode against unfinished full-block backend puts.

        ``attention_finished`` calls ``DSAKVBackend.put_blocks`` inline, then
        marks this layer/request ready. This guard is therefore a phase-order
        assertion, not an async wait. An asynchronous backend must complete the
        put before returning; otherwise this readiness table and full-cache
        block recycling contract must be replaced with completion-driven state.
        """
        if not self.kv_backend.requires_prefill_put:
            return
        layer_id = begin_batch.layer_id
        pool_indices = begin_batch.sparse_decode_guard_pool_indices_tensor.to(
            device=self.full_dump_done_by_pool.device, dtype=torch.long)
        valid_pool_mask = (
            (pool_indices >= 0)
            & (pool_indices < int(self.full_dump_done_by_pool.shape[0]))
        )
        ready_mask = torch.zeros_like(valid_pool_mask, dtype=torch.bool)
        if bool(valid_pool_mask.any().item()):
            ready_mask[valid_pool_mask] = self.full_dump_done_by_pool[
                pool_indices[valid_pool_mask], layer_id]
        if bool(ready_mask.all().item()):
            return
        first_bad = int((~ready_mask).nonzero(as_tuple=False)[0].item())
        request_id = begin_batch.sparse_decode_guard_request_ids[first_bad]
        raise RuntimeError(
            f"DSA sparse decode requires the prefill full-block backend put "
            f"to complete before shrinking full-cache blocks for req "
            f"{request_id} layer {layer_id}")

    def _build_layer_sparse_decode_batch(
        self,
        layer_name: str,
        attn_metadata,
    ) -> DSALayerSparseDecodeBatch:
        forward_batch = self.forward_sparse_decode_batch
        active_local_rows = getattr(
            forward_batch, "active_local_row_indices_tensor", None)
        if not torch.is_tensor(active_local_rows):
            raise RuntimeError(
                "DSA sparse decode batch is missing active local row indices")
        # Row-mode lookup runs over the active decode rows: dense rows keep
        # native full-cache semantics, sparse rows refresh resident metadata.
        # The true sparse subset remains in sparse_row_mask_tensor and
        # sparse_*row_indices_tensor for diagnostics and tests.
        active_local_rows = active_local_rows.to(
            device=forward_batch.resident_pool_indices_tensor.device,
            dtype=torch.long).reshape(-1)
        num_active_rows = int(active_local_rows.numel())

        layer_id = int(layer_name.split(".")[2])
        all_rows_active = (
            num_active_rows
            == int(forward_batch.resident_pool_indices_tensor.numel()))
        if all_rows_active:
            resident_pool_indices = forward_batch.resident_pool_indices_tensor
            budget_lengths = forward_batch.budget_lengths_tensor
        else:
            resident_pool_indices = (
                forward_batch.resident_pool_indices_tensor.index_select(
                    0, active_local_rows))
            budget_lengths = (
                forward_batch.budget_lengths_tensor.index_select(
                    0, active_local_rows))
        resident_view = self.get_layer_resident_resource_view_by_index(
            layer_id=layer_id,
            pool_indices=resident_pool_indices,
        )

        return DSALayerSparseDecodeBatch(
            layer_id=layer_id,
            resident_pool_indices_tensor=resident_pool_indices,
            budget_lengths_tensor=budget_lengths,
            resident_view=resident_view,
            attention_indices_width=forward_batch.attention_indices_width,
        )

    def _apply_layer_sparse_decode_batch(
        self,
        layer_batch: DSALayerSparseDecodeBatch,
        attn_metadata,
    ):
        layer_id = layer_batch.layer_id
        # Lightning Indexer returns TopK ids over the original full sequence.
        # Dumped-history ids are resolved through lookup and materialized on a
        # miss; live-tail ids map directly into the independent resident tail.
        prebuilt_attention_indices = (
            self._forward_sparse_decode_attention_indices_tensor)
        full_batch_topk = getattr(
            attn_metadata, "dsa_full_batch_selection_topk_indices", None)
        if not torch.is_tensor(full_batch_topk):
            raise RuntimeError(
                "DSA lookup resident requires full-batch "
                "lightning-indexer topk indices aligned with decode rows; "
                "the legacy sparse-only path is disabled.")
        forward_batch = self.forward_sparse_decode_batch
        (
            initial_resident_token_ids,
            locally_initialized_request_ids,
        ) = self._build_local_resident_initial_token_ids(
            layer_id=layer_id,
            forward_batch=forward_batch,
        )
        if forward_batch.batch_row_indices == list(
                range(len(forward_batch.batch_row_indices))):
            selection_topk = full_batch_topk
        else:
            selection_topk = full_batch_topk.index_select(
                0, forward_batch.batch_row_indices_tensor)
        lookup_state = self.resident_token_pool.get_layer_lookup_state(layer_id)
        lookup_result = self.ops_backend.lookup_resident_update(
            layer_id=layer_id,
            kv_backend=self.kv_backend,
            selection_topk_indices=selection_topk,
            req_pool_entries=forward_batch.resident_pool_indices_tensor,
            sparse_local_row_indices=(
                forward_batch.sparse_local_row_indices_tensor),
            selection_block_table=forward_batch.batch_hbm_block_table,
            lookup_state=lookup_state,
            resident_tokens=self._hbm_resident_tokens,
            dense_tail_starts=(
                forward_batch.dense_tail_starts_tensor),
            resident_tail_starts=(
                forward_batch.resident_tail_starts_tensor),
            attention_indices_width=layer_batch.attention_indices_width,
            prebuilt_attention_indices=prebuilt_attention_indices,
            row_modes=forward_batch.row_modes_tensor,
            lookup_init_mask=forward_batch.lookup_init_mask_tensor,
            has_lookup_init_rows=forward_batch.has_lookup_init_rows,
            initial_resident_token_ids=initial_resident_token_ids,
            maintain_seed=self._lookup_maintain_seed,
        )
        self._consume_local_prefill_layer_topk(
            layer_id=layer_id,
            request_ids=locally_initialized_request_ids,
        )
        sparse_attention_indices = lookup_result.attention_indices
        self._commit_lookup_resident_metadata(layer_batch)
        attention_indices = sparse_attention_indices
        if prebuilt_attention_indices is None:
            self._forward_sparse_decode_attention_indices_tensor = (
                attention_indices)
        attn_metadata.dsa_sparse_attention_indices = attention_indices
        return attention_indices

    def _commit_lookup_resident_metadata(
        self,
        layer_batch: DSALayerSparseDecodeBatch,
    ) -> None:
        """Mirror stable lookup occupancy into resident metadata."""
        pool_indices = layer_batch.resident_pool_indices_tensor.reshape(-1).to(
            device=layer_batch.resident_view.device,
            dtype=torch.long,
        )
        row_count = int(pool_indices.numel())
        if row_count <= 0:
            return

        row_modes = getattr(self.forward_sparse_decode_batch,
                            "row_modes_tensor", None)
        counts = layer_batch.resident_view.counts
        target_counts = torch.full(
            (row_count,),
            int(self._hbm_resident_tokens),
            dtype=counts.dtype,
            device=layer_batch.resident_view.device,
        )
        if torch.is_tensor(row_modes) and int(row_modes.numel()) == row_count:
            sparse_mask = row_modes.reshape(-1).to(
                device=layer_batch.resident_view.device,
                dtype=torch.long,
            ) == int(DSADecodeRowMode.SPARSE)
            # This function runs inside FULL graph capture/replay.  Do not use
            # .item(), boolean Python branches, or dynamic boolean indexing
            # here: those trigger D2H sync/copy and are illegal while the
            # stream is captured.  Dense rows preserve their previous resident
            # counts; sparse rows receive the stable resident count.
            current_counts = counts.index_select(0, pool_indices)
            target_counts = torch.where(
                sparse_mask,
                target_counts,
                current_counts,
            )

        counts.index_copy_(
            0,
            pool_indices,
            target_counts,
        )

    # Layer-level DSA hook before MLA/SFA.
    # Current responsibilities:
    # 1. bind this layer's cache zones for later backend put/load;
    # 2. guard sparse decode until this layer's prefill block put is ready.
    def attention_begin(self, layer_name, forward_context: ForwardContext):
        layer_id = int(layer_name.split(".")[2])
        layer_cache_zones = self.layer_cache_registry.get(layer_id)
        if layer_cache_zones is None:
            resolved_cache_zones = resolve_layer_cache_zones(layer_name,
                                                             forward_context)
            # Cache zones are worker-lifetime resources. Resolve them only on
            # first sight; later forwards use the registry fast path to avoid
            # repeated Python object traversal and tensor identity checks.
            layer_cache_zones = self.layer_cache_registry.bind_or_validate(
                layer_id, resolved_cache_zones)
            self.kv_backend.register_layer_cache(
                layer_id=layer_id,
                block_size=int(self._vllm_blk_size),
                nopek_cache=layer_cache_zones.nopek_cache_zone,
                ropek_cache=layer_cache_zones.ropek_cache_zone,
            )
        begin_batch = self._build_layer_runtime_batch(
            layer_name,
            cache_zones=layer_cache_zones,
        )
        if int(begin_batch.sparse_decode_guard_pool_indices_tensor.numel()) > 0:
            self._ensure_layer_begin_sparse_decode_dump_ready(begin_batch)

    def _put_layer_full_blocks_to_backend_batch(
        self,
        layer_batch: DSALayerRuntimeBatch,
    ) -> None:
        """Put newly completed MLA full blocks through the KV backend.

        The dump rows are built once per model forward, while the actual cache
        tensors are registered once per layer. Only after put_blocks returns do
        we mark prefill readiness for sparse decode. Decode full-block puts also
        go through this path, but the readiness bit below is only the
        prefill-to-decode phase guard.
        """
        dump_tables = layer_batch.full_block_dump_tables
        if dump_tables.request_ids:
            layer_id = layer_batch.layer_id
            self.kv_backend.put_blocks(
                layer_id=layer_id,
                request_ids=dump_tables.request_ids,
                request_pool_indices=dump_tables.request_pool_indices,
                logical_block_index_rows=dump_tables.logical_block_index_rows,
                block_key_rows=dump_tables.block_hash_rows,
                source_block_id_rows=dump_tables.block_id_rows,
                source_indexer_block_id_rows=(
                    dump_tables.indexer_block_id_rows),
                valid_token_count_rows=dump_tables.valid_token_count_rows,
            )

        self._mark_full_dump_done(
            layer_batch.prefill_done_pool_indices_tensor,
            layer_batch.layer_id,
        )

    # Layer-level DSA hook after MLA/SFA.
    # Current responsibilities:
    # 1. put this layer's prefill/decode newly-full MLA blocks into the backend;
    # 2. mark prefill put readiness for later sparse decode.
    # Token-level sparse selection/materialization is handled by after_indexer.
    def attention_finished(self, layer_name: str):
        layer_batch = self._build_layer_runtime_batch(layer_name)
        self._put_layer_full_blocks_to_backend_batch(layer_batch)

    def prepare_indexer_score_controls(self, layer_name: str, attn_metadata):
        # This hook prepares the Python-side controls around the document-level
        # dsa_compute_score_pseudo operator. The dense score tensor itself is
        # produced by the Ascend attention backend and consumed by
        # the Ascend lookup backend implementation.
        _reset_indexer_score_controls(attn_metadata)

        if not self.forward_sparse_decode_batch:
            return

        score_topk_k = self.forward_sparse_decode_batch.score_topk_k

        if score_topk_k > 0:
            setattr(attn_metadata, "dsa_score_topk_k", score_topk_k)


    # Token-level sparse IO runs after the indexer has produced dense scores.
    def after_indexer(self, layer_name: str, attn_metadata):
        if hasattr(attn_metadata, "dsa_sparse_attention_indices"):
            delattr(attn_metadata, "dsa_sparse_attention_indices")

        logger.info_once(
            "DSA sparse after_indexer entered: building the layer resident "
            "decode batch"
        )
        layer_batch = self._build_layer_sparse_decode_batch(
            layer_name, attn_metadata)
        if not layer_batch:
            return None

        logger.info_once(
            "DSA sparse layer batch ready: entering the lookup resident backend"
        )
        return self._apply_layer_sparse_decode_batch(layer_batch,
                                                     attn_metadata)

    def request_finished_in_worker(self, request_id):
        self._prefill_layer_topk.pop(request_id, None)
        self._clear_full_dump_done(request_id)
        pool_idx = int(self.resident_token_pool.get_index(request_id))
        self.kv_backend.release_request(
            request_id=request_id, request_pool_idx=pool_idx)
        self.resident_token_pool.release(request_id)
        self._pd_initialized_requests.discard(request_id)

    def request_preempted_in_worker(self, request_id):
        self._prefill_layer_topk.pop(request_id, None)
        self._clear_full_dump_done(request_id)
        pool_idx = int(self.resident_token_pool.get_index(request_id))
        self.kv_backend.release_request(
            request_id=request_id, request_pool_idx=pool_idx)
        self.resident_token_pool.release(request_id)
        self._pd_initialized_requests.discard(request_id)

    def execute_begin(self, scheduler_output: SchedulerOutput):
        pass

    def execute_finished(self):
        pass

    def _get_sparse_tail_slots_need(self, request: Request) -> int:
        total_tokens = int(request.num_tokens)
        if total_tokens <= 0:
            return 0
        full_blocks_before_tail = (total_tokens - 1) // self._vllm_blk_size
        return total_tokens - full_blocks_before_tail * self._vllm_blk_size

    def _should_preserve_sparse_tail_block(
        self,
        request: Request,
        dense_new_tokens: int,
    ) -> bool:
        previous_num_tokens = max(0, int(request.num_tokens) - int(dense_new_tokens))
        return previous_num_tokens % self._vllm_blk_size != 0

    def _release_full_blocks_except_tail(
        self,
        full_manager,
        request_id: ReqType,
        preserve_tail_block: bool,
    ) -> KVCacheBlock | None:
        req_blocks = full_manager.req_to_blocks.get(request_id)
        if not req_blocks:
            return None

        tail_block = req_blocks[-1] if preserve_tail_block else None
        full_blocks_to_release = req_blocks[:-1]
        if not preserve_tail_block:
            full_blocks_to_release = req_blocks
        full_manager.req_to_blocks[request_id] = []
        if full_blocks_to_release:
            full_manager._free_blocks_to_pool(
                reversed(full_blocks_to_release))
        full_manager.num_cached_block.pop(request_id, None)
        return tail_block

    @staticmethod
    def _append_preserved_tail_block(
        full_manager,
        request_id: ReqType,
        preserved_tail_block: KVCacheBlock | None,
    ) -> None:
        if preserved_tail_block is None:
            return
        req_blocks = full_manager.req_to_blocks[request_id]
        req_blocks.append(preserved_tail_block)

    def _allocate_pd_sparse_slots(
        self,
        kv_cache_manager: KVCacheManager,
        request: Request,
        resident_valid_seq_len: int,
        num_new_tokens: int,
        *,
        num_new_computed_tokens: int,
        num_external_computed_tokens: int,
        num_lookahead_tokens: int,
        delay_cache_blocks: bool,
        num_encoder_tokens: int,
    ) -> KVCacheBlocks | None:
        manifest = get_dsa_kvio_pd_manifest(request.kv_transfer_params)
        if manifest is None or num_external_computed_tokens <= 0:
            return None
        if resident_valid_seq_len == INVALID_SLOT:
            raise RuntimeError(
                "DSA KVIO P/D external load did not produce a sparse "
                "resident allocation plan")
        if (
            num_new_computed_tokens > 0
            or num_lookahead_tokens > 0
            or delay_cache_blocks
            or num_encoder_tokens > 0
        ):
            raise RuntimeError(
                "DSA KVIO P/D compact allocation does not support local "
                "prefix blocks, lookahead, async load, or encoder blocks")
        if num_external_computed_tokens != manifest.stored_token_count:
            raise RuntimeError(
                "DSA KVIO P/D external token count does not match manifest: "
                f"{num_external_computed_tokens} vs "
                f"{manifest.stored_token_count}")

        coordinator = kv_cache_manager.coordinator
        block_pool = kv_cache_manager.block_pool
        full_group_id = self._get_full_attention_group_id(
            kv_cache_manager.kv_cache_config)
        indexer_group_id = self._get_indexer_group_id(
            kv_cache_manager.kv_cache_config)
        full_manager = coordinator.single_type_managers[full_group_id]
        indexer_manager = coordinator.single_type_managers[indexer_group_id]

        dense_slots = min(
            int(num_external_computed_tokens) + int(num_new_tokens),
            kv_cache_manager.max_model_len,
        )
        indexer_blocks_need = indexer_manager.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=dense_slots,
            new_computed_blocks=[],
            total_computed_tokens=int(num_external_computed_tokens),
            num_tokens_main_model=dense_slots,
        )
        resident_blocks_need = full_manager.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=int(resident_valid_seq_len),
            new_computed_blocks=[],
            total_computed_tokens=int(resident_valid_seq_len),
            num_tokens_main_model=int(resident_valid_seq_len),
        )
        if (
            indexer_blocks_need
            > self._get_group_num_free_blocks(block_pool, indexer_group_id)
            or resident_blocks_need
            > self._get_group_num_free_blocks(block_pool, full_group_id)
        ):
            return None

        indexer_manager.allocate_new_blocks(
            request.request_id,
            dense_slots,
            dense_slots,
        )
        full_manager.allocate_new_blocks(
            request.request_id,
            int(resident_valid_seq_len),
            int(resident_valid_seq_len),
        )
        request.dsa_req_stage = request.dsa_next_req_stage
        request.dsa_resident_valid_seq_len = int(resident_valid_seq_len)
        request.dsa_sparse_budget_tokens = self._hbm_sparse_budget_tokens
        return KVCacheBlocks(coordinator.get_blocks(request.request_id))

    def dsa_alloc_slots_wrap(
        self,
        kv_cache_manager: KVCacheManager,
        request: Request,
        resident_valid_seq_len: int,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        def allocate_dense() -> KVCacheBlocks | None:
            request.dsa_next_req_stage = (
                ReqStage.PREFILL if request.num_output_tokens == 0
                else ReqStage.DENSE_DECODE)
            request.dsa_resident_valid_seq_len = INVALID_SLOT
            request.dsa_sparse_budget_tokens = 0
            dense_blocks = kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=num_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=delay_cache_blocks,
                num_encoder_tokens=num_encoder_tokens,
            )
            if dense_blocks is not None:
                request.dsa_req_stage = request.dsa_next_req_stage
            return dense_blocks

        if (
            num_external_computed_tokens > 0
            and get_dsa_kvio_pd_manifest(request.kv_transfer_params) is not None
        ):
            return self._allocate_pd_sparse_slots(
                kv_cache_manager,
                request,
                resident_valid_seq_len,
                num_new_tokens,
                num_new_computed_tokens=num_new_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                num_lookahead_tokens=num_lookahead_tokens,
                delay_cache_blocks=delay_cache_blocks,
                num_encoder_tokens=num_encoder_tokens,
            )

        if (request.num_computed_tokens < request.num_prompt_tokens
                or resident_valid_seq_len == INVALID_SLOT):
            return allocate_dense()

        if (num_new_computed_tokens > 0
                or num_external_computed_tokens > 0
                or num_lookahead_tokens > 0
                or delay_cache_blocks
                or num_encoder_tokens > 0):
            return allocate_dense()

        else:
            coordinator = kv_cache_manager.coordinator
            block_pool = kv_cache_manager.block_pool

            # ENTER_SPARSE_DECODE shrinks the full/MLA table to sparse-budget
            # blocks plus an optional unfilled tail block. This covers both the
            # old long-prompt first decode and the short-prompt long-decode
            # transition once the sequence crosses the sparse threshold.
            full_group_id = self._get_full_attention_group_id(
                kv_cache_manager.kv_cache_config)
            indexer_group_id = self._get_indexer_group_id(
                kv_cache_manager.kv_cache_config)
            full_manager = coordinator.single_type_managers[full_group_id]
            indexer_manager = coordinator.single_type_managers[indexer_group_id]
            dense_computed_tokens = (
                request.num_computed_tokens
                + max(0, int(num_new_computed_tokens))
                + max(0, int(num_external_computed_tokens)))
            dense_num_tokens_need_slot = min(
                dense_computed_tokens
                + max(0, int(num_new_tokens))
                + max(0, int(num_lookahead_tokens)),
                kv_cache_manager.max_model_len,
            )
            indexer_blocks_to_allocate = (
                indexer_manager.get_num_blocks_to_allocate(
                    request_id=request.request_id,
                    num_tokens=dense_num_tokens_need_slot,
                    new_computed_blocks=[],
                    total_computed_tokens=dense_computed_tokens,
                    num_tokens_main_model=dense_num_tokens_need_slot,
                ))
            if indexer_blocks_to_allocate > self._get_group_num_free_blocks(
                    block_pool, indexer_group_id):
                return None

            req_stage = request.dsa_next_req_stage
            reset_full_cache = req_stage.is_enter_sparse_decode
            preserved_tail_block = None
            sparse_budget_slots = resident_valid_seq_len
            if reset_full_cache:
                if request.num_computed_tokens < request.num_prompt_tokens:
                    return allocate_dense()
                tail_slots_need = self._get_sparse_tail_slots_need(request)
                preserve_tail_block = self._should_preserve_sparse_tail_block(
                    request, num_new_tokens)
                existing_full_blocks = full_manager.req_to_blocks.get(
                    request.request_id, [])
                will_preserve_tail = (
                    preserve_tail_block and bool(existing_full_blocks))
                sparse_budget_slots = (
                    max(0, resident_valid_seq_len - tail_slots_need)
                    if will_preserve_tail
                    else resident_valid_seq_len)
                releasable_full_blocks = max(
                    0,
                    len(existing_full_blocks)
                    - (1 if will_preserve_tail else 0),
                )
                sparse_budget_blocks_need = (
                    (sparse_budget_slots + full_manager.block_size - 1)
                    // full_manager.block_size
                    if sparse_budget_slots > 0 else 0)
                full_blocks_available_after_release = (
                    self._get_group_num_free_blocks(
                        block_pool, full_group_id)
                    + releasable_full_blocks)
                if (sparse_budget_blocks_need
                        > full_blocks_available_after_release):
                    return None
                preserved_tail_block = self._release_full_blocks_except_tail(
                    full_manager, request.request_id, preserve_tail_block)

            num_blocks_to_allocate = full_manager.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=sparse_budget_slots,
                new_computed_blocks=[],
                total_computed_tokens=sparse_budget_slots,
                num_tokens_main_model=sparse_budget_slots,
            )
            has_enough_blocks = (
                num_blocks_to_allocate <= self._get_group_num_free_blocks(
                    block_pool, full_group_id))
            if not has_enough_blocks:
                if reset_full_cache:
                    raise RuntimeError(
                        "DSA sparse allocation capacity precheck passed but "
                        "post-release capacity check failed")
                self._append_preserved_tail_block(
                    full_manager, request.request_id, preserved_tail_block)
                return None
            full_manager.allocate_new_blocks(
                request.request_id,
                sparse_budget_slots,
                sparse_budget_slots,
            )
            self._append_preserved_tail_block(
                full_manager, request.request_id, preserved_tail_block)
            # Indexer cache is the dense selector plane. It must keep a full
            # block table for the original sequence in HBM and must not follow
            # the sparse full/MLA table shrink/replace policy.
            indexer_manager.allocate_new_blocks(
                request.request_id,
                dense_num_tokens_need_slot,
                dense_num_tokens_need_slot,
            )
            request.dsa_req_stage = req_stage
            request.dsa_next_req_stage = req_stage
            request.dsa_resident_valid_seq_len = resident_valid_seq_len
            return KVCacheBlocks(coordinator.get_blocks(request.request_id))


    def plan_decode_resident_slots(
        self,
        request: Request,
        num_external_computed_tokens: int = 0,
    ):
        # This scheduler-side planner is the single stage-advance point for DSA
        # cache layout. It both returns the resident MLA/full-cache slot count
        # for sparse decode and writes the request stage metadata consumed by
        # worker hooks. Keep this state transition out of layer-wise code.
        previous_stage = request.dsa_req_stage
        dense_stage = (
            ReqStage.PREFILL if request.num_output_tokens == 0
            else ReqStage.DENSE_DECODE)
        request.dsa_next_req_stage = dense_stage
        request.dsa_resident_valid_seq_len = INVALID_SLOT
        request.dsa_sparse_budget_tokens = 0
        if not self._is_sparse_cache_enabled():
            return INVALID_SLOT
        manifest = get_dsa_kvio_pd_manifest(request.kv_transfer_params)
        if manifest is not None and int(num_external_computed_tokens) > 0:
            manifest.validate()
            if int(num_external_computed_tokens) != manifest.stored_token_count:
                raise RuntimeError(
                    "DSA KVIO P/D scheduler token count does not match its "
                    f"manifest: {num_external_computed_tokens} vs "
                    f"{manifest.stored_token_count}")
            if request.num_tokens <= self._enable_dsa_prompt_len:
                raise ValueError(
                    "DSA KVIO P/D compact handoff requires a request longer "
                    f"than {self._enable_dsa_prompt_len} tokens, got "
                    f"{request.num_tokens}")
            candidate_full_blocks = (
                manifest.stored_token_count // self._vllm_blk_size)
            tail_slots_need = (
                request.num_tokens
                - candidate_full_blocks * self._vllm_blk_size)
            return self._plan_sparse_decode_resident_slots(
                request=request,
                candidate_full_blocks=candidate_full_blocks,
                tail_slots_need=tail_slots_need,
                previous_stage=ReqStage.PREFILL,
            )
        if request.num_computed_tokens < request.num_prompt_tokens:
            return INVALID_SLOT
        if request.num_output_tokens == 0:  # prefill/chunked_prefill
            return INVALID_SLOT
        if (request.spec_token_ids or request.num_output_placeholders
                or request.has_encoder_inputs):
            return INVALID_SLOT
        if request.num_tokens <= self._enable_dsa_prompt_len:
            return INVALID_SLOT

        block_size = self._vllm_blk_size
        total_tokens = request.num_tokens
        full_blocks_before_tail = (total_tokens - 1) // block_size
        tail_slots_need = total_tokens - full_blocks_before_tail * block_size
        if full_blocks_before_tail <= 0:
            return INVALID_SLOT
        return self._plan_sparse_decode_resident_slots(
            request=request,
            candidate_full_blocks=full_blocks_before_tail,
            tail_slots_need=tail_slots_need,
            previous_stage=previous_stage,
        )

    def _plan_sparse_decode_resident_slots(
            self,
            request: Request,
            candidate_full_blocks: int,
            tail_slots_need: int,
            previous_stage: ReqStage,
    ) -> int:
        block_size = self._vllm_blk_size
        total_tokens = int(request.num_tokens)
        candidate_tokens = candidate_full_blocks * block_size
        if candidate_tokens <= 0:
            return INVALID_SLOT

        sparse_budget_tokens = self._get_fixed_sparse_budget_tokens(
            candidate_tokens)
        if sparse_budget_tokens <= 0:
            return INVALID_SLOT

        resident_valid_seq_len = (
            self._lookup_total_slot_tokens + tail_slots_need)
        next_stage = (
            ReqStage.SPARSE_DECODE
            if previous_stage.is_sparse_decode
            else ReqStage.ENTER_SPARSE_DECODE)
        request.dsa_next_req_stage = next_stage
        if next_stage.is_enter_sparse_decode:
            logger.info(
                "========== DSA DECODE REACHED SPARSE THRESHOLD =========="
                " req_id=%s prompt_tokens=%s output_tokens=%s total_tokens=%s "
                "computed_tokens=%s candidate_full_blocks=%s tail_slots=%s "
                "sparse_budget=%s resident_valid_seq_len=%s block_size=%s "
                "sparse_threshold=%s",
                request.request_id,
                request.num_prompt_tokens,
                request.num_output_tokens,
                total_tokens,
                request.num_computed_tokens,
                candidate_full_blocks,
                tail_slots_need,
                sparse_budget_tokens,
                resident_valid_seq_len,
                block_size,
                self._enable_dsa_prompt_len,
            )
        request.dsa_sparse_budget_tokens = sparse_budget_tokens
        request.dsa_resident_valid_seq_len = resident_valid_seq_len
        return request.dsa_resident_valid_seq_len
