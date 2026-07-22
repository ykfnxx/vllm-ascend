#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import json
import math
import re
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from vllm.logger import logger

HIXL_KV_MAX_BATCH_ENTRIES = 110
HIXL_KV_SQE_DWORD_COUNT = 16
HIXL_KV_BATCH_ENTRY_DWORD_COUNT = 9
HIXL_KV_KEY_PREFIX = 0x4849584C4B455931
HIXL_KV_KEY_BYTES = 16
HIXL_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class HixlBackendConfig:
    npu_phy_dev: int
    npu_ip: str
    kernel_json: str
    ssu_phy_dev: int | None = None
    ssu_ip: str | None = None
    source_mode: str = "external"
    coord_dir: str = "/tmp/dsa_ssu_remote_load"
    protocol: str = "roce"
    port: int = 16666
    channel_count: int = 7
    cache_size: int = 2048
    offload_block_size: int = 128
    mailbox_bytes: int = 1 << 20
    ring_slots: int = 256
    timeout_s: float = 120.0
    verify_timeout_s: float = 120.0
    poll_interval_us: int = 10
    launch_timeout_ms: int = 120_000
    entry_timeout_us: int = 0
    desired_shards: int = 7
    aicpu_microbatch_size: int = 12
    parallel_assemble_sqe: bool = True
    send_batch_groups: int = 1
    poll_terminal_cqe_fast: bool = True

    @classmethod
    def from_json(cls, path: str) -> "HixlBackendConfig":
        config_path = Path(path)
        if not config_path.is_file():
            raise RuntimeError(
                "HIXL DMP backend configuration is missing: "
                f"{config_path}. Keep VLLM_ASCEND_DMP_KV_BACKEND=local or "
                "provide VLLM_ASCEND_DMP_HIXL_CONFIG."
            )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        required = ("npu_phy_dev", "npu_ip", "kernel_json")
        missing = [name for name in required if name not in payload]
        if missing:
            raise RuntimeError(f"HIXL DMP configuration {config_path} is missing: " + ", ".join(missing))
        return cls(**payload)


@dataclass
class HixlLayerState:
    cached_token_slots: torch.Tensor
    slot_token_ids: torch.Tensor
    next_evict_slot: torch.Tensor
    visit_generation: torch.Tensor
    request_signature: torch.Tensor
    previous_seq_lens: torch.Tensor


@dataclass
class HixlWorkspaceState:
    layer_id: int
    session: Any
    req_pool_entries: torch.Tensor
    hit_token_slots: torch.Tensor
    miss_token_ids: torch.Tensor
    miss_token_slots: torch.Tensor
    success_count: torch.Tensor
    success_token_slots: torch.Tensor
    debug_info: torch.Tensor
    outputs: Any
    result: Any = None


def _request_payload_bytes(entry_count: int) -> int:
    base = (HIXL_KV_SQE_DWORD_COUNT + int(entry_count) * HIXL_KV_BATCH_ENTRY_DWORD_COUNT) * 4
    return base + 96 + int(entry_count) * 88


def _cqe_bytes(entry_count: int) -> int:
    return (4 + (int(entry_count) + 7) // 8) * 4


def _channel_partitions(
    total_slots: int,
    slots_per_sqe: int,
    channel_count: int,
) -> tuple[list[int], list[int], list[int]]:
    pending_counts = []
    sqe_counts = []
    sqe_entry_counts = []
    for channel_index in range(channel_count):
        begin = total_slots * channel_index // channel_count
        end = total_slots * (channel_index + 1) // channel_count
        pending = end - begin
        sqes = 0 if pending == 0 else math.ceil(pending / slots_per_sqe)
        pending_counts.append(pending)
        sqe_counts.append(sqes)
        for sqe_index in range(sqes):
            slots = min(slots_per_sqe, pending - sqe_index * slots_per_sqe)
            sqe_entry_counts.append(slots * 2)
    return pending_counts, sqe_counts, sqe_entry_counts


def _ready_group_sqe_capacity(
    batch_size: int,
    topk: int,
    slots_per_sqe: int,
    channel_count: int,
    group_size: int,
) -> int:
    channel_sqes = [0] * channel_count
    for batch_begin in range(0, batch_size, group_size):
        batch_end = min(batch_begin + group_size, batch_size)
        group_batches = batch_end - batch_begin
        worker_count = min(channel_count, group_batches)
        group_offset = batch_begin % worker_count
        sqes_per_row = math.ceil(topk / slots_per_sqe)
        for local_row in range(group_batches):
            channel = (local_row + group_offset) % worker_count
            channel_sqes[channel] += sqes_per_row
    return max(channel_sqes) * channel_count


def _make_offload_block_keys(
    pool_size: int,
    source_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    keys = torch.empty((pool_size, source_blocks, HIXL_KV_KEY_BYTES), dtype=torch.uint8)
    for pool_entry in range(pool_size):
        for source_block in range(source_blocks):
            value = struct.pack(
                "<QQ",
                HIXL_KV_KEY_PREFIX ^ pool_entry,
                (pool_entry << 32) | source_block,
            )
            keys[pool_entry, source_block] = torch.tensor(list(value), dtype=torch.uint8)
    return keys.to(device=device)


class HixlDualAttentionBackend:
    """Experimental IndexerUpdate/HIXL backend for DMP Dual-Attention.

    The peer SSU must already contain CKV/KPE data using the same request-pool,
    block-key, layer, and token layout. The repository SSU emulator uses
    synthetic data and is suitable for operator/graph smoke tests only.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        block_size: int,
        max_microbatch_tokens: int,
        max_seq_len: int,
        topk: int,
        kv_lora_rank: int,
        rope_dim: int,
        num_layers: int,
        config: HixlBackendConfig,
    ) -> None:
        if max_microbatch_tokens <= 0 or max_seq_len <= 0 or topk <= 0:
            raise ValueError("HIXL microbatch capacity, max sequence length, and top-k must all be positive")
        if num_layers <= 0:
            raise ValueError("HIXL num_layers must be positive")
        if not 1 <= config.channel_count <= 7:
            raise ValueError("HIXL channel_count must be between 1 and 7")
        if config.aicpu_microbatch_size <= 0:
            raise ValueError("HIXL aicpu_microbatch_size must be positive")
        if config.offload_block_size <= 0:
            raise ValueError("HIXL offload_block_size must be positive")
        if config.source_mode not in ("external", "synthetic"):
            raise ValueError(f"HIXL source_mode must be 'external' or 'synthetic', got {config.source_mode!r}")
        if config.cache_size < topk:
            raise ValueError(f"HIXL cache_size={config.cache_size} must be >= topk={topk}")
        if config.cache_size % block_size != 0:
            raise ValueError(
                "HIXL cache_size must be divisible by the vLLM cache block "
                f"size: {config.cache_size} % {block_size} != 0"
            )
        kernel_json = Path(config.kernel_json)
        if not kernel_json.is_file():
            raise RuntimeError(f"HIXL kernel JSON does not exist: {kernel_json}")

        try:
            import dsa_offload_overlap_custom_ops as dsa_ops
        except ImportError as exc:
            raise RuntimeError(
                "HIXL DMP backend requires dsa_offload_overlap_custom_ops from the hixl-li-ready-smoke branch"
            ) from exc
        if dsa_ops.custom_ops_hixl_lib is None:
            raise RuntimeError("The HIXL torch extension is unavailable")

        self.device = device
        self.dtype = dtype
        self.block_size = int(block_size)
        self.max_microbatch_tokens = int(max_microbatch_tokens)
        self.pool_size = 2 * self.max_microbatch_tokens
        self.max_seq_len = int(max_seq_len)
        self.topk = int(topk)
        self.kv_lora_rank = int(kv_lora_rank)
        self.rope_dim = int(rope_dim)
        self.num_layers = int(num_layers)
        self.config = config
        self.cache_size = int(config.cache_size)
        self.dsa_ops = dsa_ops
        self._layer_states: dict[int, HixlLayerState] = {}
        self._sessions: dict[int, Any] = {}

        element_size = torch.empty((), dtype=dtype).element_size()
        self.ckv_bytes_per_slot = self.kv_lora_rank * element_size
        self.kpe_bytes_per_slot = self.rope_dim * element_size
        self.layer_slot_count = self.pool_size * int(config.cache_size)
        self.ckv_layer_bytes = self.layer_slot_count * self.ckv_bytes_per_slot
        self.kpe_layer_bytes = self.layer_slot_count * self.kpe_bytes_per_slot
        self.layer_stride = self.ckv_layer_bytes + self.kpe_layer_bytes

        source_blocks = math.ceil(self.max_seq_len / config.offload_block_size)
        offload_block_keys = _make_offload_block_keys(self.pool_size, source_blocks, device)
        entries_per_slot = 2
        slots_per_sqe = HIXL_KV_MAX_BATCH_ENTRIES // entries_per_slot
        total_slots = self.max_microbatch_tokens * self.topk
        pending_counts, sqe_counts, sqe_entry_counts = _channel_partitions(
            total_slots, slots_per_sqe, config.channel_count
        )
        sqe_capacity = _ready_group_sqe_capacity(
            self.max_microbatch_tokens,
            self.topk,
            slots_per_sqe,
            int(config.channel_count),
            int(config.aicpu_microbatch_size),
        )
        max_entry_count = slots_per_sqe * entries_per_slot
        success_count = torch.empty(self.max_microbatch_tokens, dtype=torch.int32, device=device)
        success_slots = torch.empty(
            (self.max_microbatch_tokens, self.topk),
            dtype=torch.int32,
            device=device,
        )
        hixl = dsa_ops.custom_ops_hixl_lib
        endpoint = dsa_ops.HcommEndpointConfig(
            phy_dev=int(config.npu_phy_dev),
            ip=str(config.npu_ip),
            protocol=str(config.protocol),
            port=int(config.port),
            timeout_s=float(config.timeout_s),
            mailbox_bytes=int(config.mailbox_bytes),
            ring_slots=int(config.ring_slots),
            channel_count=int(config.channel_count),
            request_capacity=_request_payload_bytes(max_entry_count),
            max_request_payload_bytes=_request_payload_bytes(max_entry_count),
        )
        coord_dir = Path(config.coord_dir)
        paths = dsa_ops.HixlRendezvousPaths(
            local=coord_dir / "npu.json",
            peer=coord_dir / "ssu.json",
        )
        setup_config = dsa_ops.HixlRemoteLoadSetupConfig(
            endpoint=endpoint,
            kernel_json=str(kernel_json),
            batch_size=self.pool_size,
            cache_size=int(config.cache_size),
            bytes_per_entry=self.ckv_bytes_per_slot,
            kpe_bytes_per_slot=self.kpe_bytes_per_slot,
            offload_block_size=int(config.offload_block_size),
            cqe_entry_bytes=_cqe_bytes(max_entry_count),
            request_payload_bytes=_request_payload_bytes(max_entry_count),
            hcomm_ring_header_bytes=int(hixl.HCOMM_SEND_RECV_RING_HEADER_SIZE),
            channel_pending_counts=pending_counts,
            channel_sqe_counts=sqe_counts,
            sqe_entry_counts=sqe_entry_counts,
            sqe_capacity=max(sum(sqe_counts), sqe_capacity),
            parallel_assemble_sqe=bool(config.parallel_assemble_sqe),
            send_batch_groups=int(config.send_batch_groups),
            entry_timeout_us=int(config.entry_timeout_us),
            launch_timeout_ms=int(config.launch_timeout_ms),
            verify_timeout_s=float(config.verify_timeout_s),
            poll_interval_s=float(config.poll_interval_us) / 1_000_000.0,
            poll_terminal_cqe_fast=bool(config.poll_terminal_cqe_fast),
            ckv_base_offset=0,
            kpe_base_offset=self.ckv_layer_bytes,
            ckv_source_base_offset=0,
            kpe_source_base_offset=(self.num_layers * int(config.offload_block_size) * self.ckv_bytes_per_slot),
            data_bytes_override=self.num_layers * self.layer_stride,
        )
        self.client = dsa_ops.setup_hcomm_remote_load(
            torch,
            hixl,
            setup_config,
            paths,
            device=device,
            offload_block_keys=offload_block_keys,
            success_count=success_count,
            success_token_slots=success_slots,
        )

        source_bytes = (
            self.pool_size * self.max_seq_len * self.num_layers * (self.ckv_bytes_per_slot + self.kpe_bytes_per_slot)
        )
        logger.warning(
            "Enabled experimental HIXL DMP backend: pool=%d cache=%d "
            "destination=%.2f GiB estimated_full_source=%.2f GiB "
            "source_mode=%s. The NPU SSU emulator is intended for reduced "
            "smoke configurations.",
            self.pool_size,
            config.cache_size,
            (self.num_layers * self.layer_stride) / (1 << 30),
            source_bytes / (1 << 30),
            config.source_mode,
        )
        if config.source_mode == "synthetic":
            logger.warning(
                "HIXL DMP is reading synthetic SSU data; graph/transport can "
                "be tested, but generated model outputs are not meaningful."
            )
        # State and per-layer sessions must exist before ACL graph capture;
        # replay cannot execute Python allocation or HCOMM setup.
        for layer_id in range(self.num_layers):
            self._layer_state(layer_id)
            self._session(layer_id)

    @classmethod
    def from_json(cls, config_path: str, **kwargs) -> "HixlDualAttentionBackend":
        return cls(config=HixlBackendConfig.from_json(config_path), **kwargs)

    def _layer_id(self, layer_name: str) -> int:
        match = HIXL_LAYER_PATTERN.search(layer_name)
        if match is None:
            raise RuntimeError(f"Cannot extract layer id from HIXL DMP layer name: {layer_name}")
        layer_id = int(match.group(1))
        if layer_id < 0 or layer_id >= self.num_layers:
            raise RuntimeError(f"HIXL layer id is out of range: {layer_id}")
        return layer_id

    def cache_tensors(
        self,
        layer_name: str,
        kv_dtype: torch.dtype,
        rope_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_dtype != self.dtype or rope_dtype != self.dtype:
            raise RuntimeError(
                f"HIXL destination cache dtype mismatch: configured={self.dtype}, kv={kv_dtype}, rope={rope_dtype}"
            )
        layer_id = self._layer_id(layer_name)
        layer_base = layer_id * self.layer_stride
        ckv_bytes = self.client.data.narrow(0, layer_base, self.ckv_layer_bytes)
        kpe_bytes = self.client.data.narrow(
            0,
            layer_base + self.ckv_layer_bytes,
            self.kpe_layer_bytes,
        )
        block_count = self.layer_slot_count // self.block_size
        ckv = ckv_bytes.view(self.dtype).view(block_count, self.block_size, self.kv_lora_rank)
        kpe = kpe_bytes.view(self.dtype).view(block_count, self.block_size, self.rope_dim)
        return ckv, kpe

    def _layer_state(self, layer_id: int) -> HixlLayerState:
        state = self._layer_states.get(layer_id)
        if state is None:
            state = HixlLayerState(
                cached_token_slots=torch.full(
                    (self.pool_size, self.max_seq_len),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                slot_token_ids=torch.full(
                    (self.pool_size, self.config.cache_size),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                next_evict_slot=torch.zeros(self.pool_size, dtype=torch.int32, device=self.device),
                visit_generation=torch.zeros(
                    (self.pool_size, self.max_seq_len),
                    dtype=torch.bool,
                    device=self.device,
                ),
                request_signature=torch.full((self.pool_size,), -1, dtype=torch.int32, device=self.device),
                previous_seq_lens=torch.full((self.pool_size,), -1, dtype=torch.int32, device=self.device),
            )
            self._layer_states[layer_id] = state
        return state

    def _session(self, layer_id: int):
        session = self._sessions.get(layer_id)
        if session is None:
            base = self.client.loader.resources
            layer_base = layer_id * self.layer_stride
            resources = replace(
                base,
                ckv_base_offset=layer_base,
                kpe_base_offset=layer_base + self.ckv_layer_bytes,
            )
            loader = self.dsa_ops.HixlScatterCopyLoader(resources)
            session = self.dsa_ops.HixlScatterCopyRemoteLoadSession(
                [loader],
                poll_results=[self.client.poll_result],
                offload_store=self.client.offload_store,
                success_counts=[self.client.success_count],
                success_token_slots=[self.client.success_token_slots],
                entries_per_slot=2,
                poll_timeout_us=max(1, int(self.config.verify_timeout_s * 1_000_000)),
                poll_interval_us=max(0, int(self.config.poll_interval_us)),
                launch_timeout_ms=int(self.config.launch_timeout_ms),
            )
            self._sessions[layer_id] = session
            self.client.keepalive.extend([resources, loader, session])
        return session

    def prepare_workspace(
        self,
        layer_name: str,
        workspace: Any,
        microbatch_idx: int,
    ) -> HixlWorkspaceState:
        layer_id = self._layer_id(layer_name)
        token_count = int(workspace.hit_sparse_indices.shape[0])
        topk = int(workspace.hit_sparse_indices.shape[2])
        row_offset = int(microbatch_idx) * self.max_microbatch_tokens
        req_pool_entries = torch.arange(
            row_offset,
            row_offset + token_count,
            dtype=torch.int32,
            device=self.device,
        )
        shape = (token_count, topk)
        outputs = self.dsa_ops.IndexerUpdateHixlLoadOutputs(
            hit_count=workspace.hit_count,
            hit_token_slots=torch.empty(shape, dtype=torch.int32, device=self.device),
            miss_count=workspace.miss_count,
            miss_token_ids=torch.empty(shape, dtype=torch.int32, device=self.device),
            miss_token_slots=torch.empty(shape, dtype=torch.int32, device=self.device),
            success_count=torch.empty((token_count,), dtype=torch.int32, device=self.device),
            success_token_slots=torch.empty(shape, dtype=torch.int32, device=self.device),
            hit_sfa_indices=workspace.hit_sparse_indices.squeeze(1),
            success_sfa_indices=workspace.miss_insert_indices.squeeze(1),
            debug_info=torch.empty(24, dtype=torch.int32, device=self.device),
        )
        return HixlWorkspaceState(
            layer_id=layer_id,
            session=self._session(layer_id),
            req_pool_entries=req_pool_entries,
            hit_token_slots=outputs.hit_token_slots,
            miss_token_ids=outputs.miss_token_ids,
            miss_token_slots=outputs.miss_token_slots,
            success_count=outputs.success_count,
            success_token_slots=outputs.success_token_slots,
            debug_info=outputs.debug_info,
            outputs=outputs,
        )

    def select(
        self,
        layer_name: str,
        microbatch_idx: int,
        workspace: Any,
        topk_indices: torch.Tensor,
        attn_metadata: Any,
    ) -> None:
        layer_id = self._layer_id(layer_name)
        state = self._layer_state(layer_id)
        backend_state = workspace.backend_state
        if not isinstance(backend_state, HixlWorkspaceState):
            raise RuntimeError("HIXL workspace state is not initialized")
        token_count = int(topk_indices.shape[0])
        if token_count > self.max_microbatch_tokens:
            raise RuntimeError(
                "HIXL DMP microbatch exceeds configured capacity: "
                f"tokens={token_count}, capacity={self.max_microbatch_tokens}"
            )
        if int(topk_indices.shape[2]) != self.topk:
            raise RuntimeError(
                "HIXL DMP top-k width changed after initialization: "
                f"got={topk_indices.shape[2]}, configured={self.topk}"
            )
        if backend_state.layer_id != layer_id:
            raise RuntimeError("HIXL workspace was reused by a different model layer")
        row_offset = int(microbatch_idx) * self.max_microbatch_tokens
        row_end = row_offset + token_count
        cached_rows = state.cached_token_slots[row_offset:row_end]
        owner_rows = state.slot_token_ids[row_offset:row_end]
        next_rows = state.next_evict_slot[row_offset:row_end]
        visit_rows = state.visit_generation[row_offset:row_end]
        signature_rows = state.request_signature[row_offset:row_end]
        previous_rows = state.previous_seq_lens[row_offset:row_end]
        seq_lens = attn_metadata.seq_lens.to(torch.int32)
        signatures = attn_metadata.block_table[:, 0].to(torch.int32)
        reassigned = (signatures != signature_rows) | (seq_lens <= previous_rows)
        cached_rows.masked_fill_(reassigned.view(-1, 1), -1)
        owner_rows.masked_fill_(reassigned.view(-1, 1), -1)
        next_rows.masked_fill_(reassigned, 0)
        signature_rows.copy_(signatures)
        previous_rows.copy_(seq_lens)

        # The current fused ABI accepts generation as a host scalar. Clearing
        # the batch rows makes repeated graph replay correct until the ABI gains
        # a device-side generation counter.
        visit_rows.zero_()
        workspace.hit_sparse_indices.fill_(-1)
        workspace.miss_insert_indices.fill_(-1)
        backend_state.success_count.zero_()
        backend_state.success_token_slots.fill_(-1)

        topk_2d = topk_indices.squeeze(1)
        backend_state.result = self.dsa_ops.indexer_update_hixl_load(
            backend_state.session,
            topk_2d,
            seq_lens,
            backend_state.req_pool_entries,
            state.cached_token_slots,
            state.slot_token_ids,
            state.next_evict_slot,
            state.visit_generation,
            max_seq_len=self.max_seq_len,
            cache_size=int(self.config.cache_size),
            pool_size=self.pool_size,
            generation=1,
            desired_shards=int(self.config.desired_shards),
            layer_id=layer_id,
            enable_hixl_load=True,
            aicpu_microbatch_size=int(self.config.aicpu_microbatch_size),
            outputs=backend_state.outputs,
        )
        workspace.selected_actual_seq.fill_(int(self.config.cache_size))
        workspace.selection_kv_actual_seq.fill_(int(self.config.cache_size))
        workspace.hit_attention_out = None

    def gather(self, workspace: Any) -> None:
        backend_state = workspace.backend_state
        if not isinstance(backend_state, HixlWorkspaceState):
            raise RuntimeError("HIXL workspace state is not initialized")
        if backend_state.result is None:
            raise RuntimeError("HIXL submit must run before poll")
        result = backend_state.result
        if result.handle is not None:
            backend_state.session.poll_wait(result.handle)
