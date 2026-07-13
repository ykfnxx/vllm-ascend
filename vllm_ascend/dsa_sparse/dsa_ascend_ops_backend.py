"""DSA 稀疏卸载的昇腾算子后端封装。

本文件负责把 Python 侧已经 tensor 化的 DSA forward/layer 元数据传给
昇腾自定义算子，当前核心是 gather-selection KV cache 路径。它只处理
算子入参整理、trace 打点和返回值契约，不负责请求状态推进、DRAM/HBM
资源分配，也不决定某个 batch 是否应该进入稀疏 decode。
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from vllm.logger import logger

from vllm_ascend.dsa_sparse.dsa_trace import (
    DSA_TRACE_POINT_GATHER_SELECTION,
    DSA_TRACE_POINT_GATHER_SELECTION_STATS,
    dsa_trace_enabled,
    dsa_trace_sync_enabled,
)


class DSAGatherSelectionOutput(NamedTuple):
    """Output contract for the Ascend gather-selection DSA path."""

    attention_indices: torch.Tensor
    resident_update_committed: bool = True


def _tensor_brief(tensor: torch.Tensor | None) -> dict:
    if not torch.is_tensor(tensor):
        return {"type": type(tensor).__name__}
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": tuple(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
    }


def _tensor_sample(tensor: torch.Tensor | None, limit: int = 8):
    if not torch.is_tensor(tensor):
        return None
    try:
        flat = tensor.detach().reshape(-1)
        if int(flat.numel()) == 0:
            return []
        return flat[:min(limit, int(flat.numel()))].cpu().tolist()
    except Exception as exc:  # pragma: no cover - debug-only best effort.
        return f"<sample unavailable: {type(exc).__name__}: {exc}>"


def _dsa_gs_backend_sync(stage: str, layer_id: int | None,
                         context: dict) -> None:
    try:
        torch.npu.current_stream().synchronize()
    except Exception:
        logger.exception("[DSA GS backend sync failed] stage=%s layer=%s "
                         "context=%s", stage, layer_id, context)
        raise


def _build_gather_selection_overlap_stats(
    *,
    layer_id: int,
    topk: torch.Tensor,
    status: torch.Tensor,
    req_pool_entries: torch.Tensor,
    row_modes: torch.Tensor,
    candidate_lens: torch.Tensor,
) -> dict:
    """Build debug-only GS hit/miss stats from the exact kernel inputs.

    统计发生在 GS op 调用前，此时 ``status`` 仍是上一轮 resident 状态：
    status[pool_idx, ..., slot] 记录 resident slot 当前存放的原始 token id。
    因此 topk 与 status 的集合交集就是本轮 GS 的 HBM hit，差集就是需要
    从 full/DRAM cache 换入的 miss。这个函数会 D2H，同步开销很大，只能在
    trace/debug 下使用，不能用于正式性能数据。
    """
    topk_cpu = topk.detach().reshape(int(topk.shape[0]), -1).cpu()
    req_pool_cpu = req_pool_entries.detach().reshape(-1).to(
        dtype=torch.long).cpu()
    row_modes_cpu = row_modes.detach().reshape(-1).cpu()
    candidate_lens_cpu = candidate_lens.detach().reshape(-1).cpu()
    status_rows_cpu = status.index_select(
        0, req_pool_entries.reshape(-1).to(dtype=torch.long)).detach()
    status_rows_cpu = status_rows_cpu.reshape(
        int(status_rows_cpu.shape[0]), -1).cpu()
    topk_width = int(topk_cpu.shape[1])
    status_topk_cpu = status_rows_cpu[:, :topk_width]

    mode_values = row_modes_cpu.tolist()
    mode_counts = {
        "pad": sum(1 for mode in mode_values if int(mode) == 0),
        "dense": sum(1 for mode in mode_values if int(mode) == 1),
        "sparse": sum(1 for mode in mode_values if int(mode) == 2),
        "unknown": sum(1 for mode in mode_values
                       if int(mode) not in (0, 1, 2)),
    }
    rows: list[dict] = []
    total_valid = 0
    total_hits = 0
    total_misses = 0
    for row_idx, mode in enumerate(mode_values):
        if int(mode) != 2:
            continue
        candidate_len = int(candidate_lens_cpu[row_idx].item())
        topk_values = [
            int(value) for value in topk_cpu[row_idx].tolist()
            if 0 <= int(value) < candidate_len
        ]
        status_values = {
            int(value) for value in status_topk_cpu[row_idx].tolist()
            if int(value) >= 0
        }
        hit_count = sum(1 for value in topk_values
                        if value in status_values)
        valid_count = len(topk_values)
        miss_count = valid_count - hit_count
        total_valid += valid_count
        total_hits += hit_count
        total_misses += miss_count
        rows.append({
            "row": int(row_idx),
            "pool_idx": int(req_pool_cpu[row_idx].item()),
            "candidate_len": candidate_len,
            "valid_topk": int(valid_count),
            "status_valid": int(len(status_values)),
            "hits": int(hit_count),
            "misses": int(miss_count),
            "overlap": (float(hit_count) / float(valid_count)
                        if valid_count else 0.0),
            "miss_rate": (float(miss_count) / float(valid_count)
                          if valid_count else 0.0),
        })

    return {
        "layer_id": int(layer_id),
        "rows": int(topk_cpu.shape[0]),
        "topk": int(topk_width),
        "mode_counts": mode_counts,
        "sparse_rows": rows,
        "total_valid_topk": int(total_valid),
        "total_hits": int(total_hits),
        "total_misses": int(total_misses),
        "overlap": (float(total_hits) / float(total_valid)
                    if total_valid else 0.0),
        "miss_rate": (float(total_misses) / float(total_valid)
                      if total_valid else 0.0),
    }


class AscendDSAOpsBackend:
    """Ascend gather-selection backend for DSA sparse-cache offload."""

    @staticmethod
    def _squeeze_cache_head_dim(cache: torch.Tensor | None,
                                name: str) -> torch.Tensor:
        if not torch.is_tensor(cache):
            raise ValueError(f"{name} is required for DSA gather-selection")
        if cache.ndim == 4 and int(cache.shape[2]) == 1:
            return cache.squeeze(2)
        if cache.ndim == 3:
            return cache
        raise ValueError(
            f"{name} must have shape [blocks, block, 1, dim] or "
            f"[blocks, block, dim], got {tuple(cache.shape)}")

    @staticmethod
    def _normalize_selection_topk_indices(
        selection_topk_indices: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        topk = selection_topk_indices.to(device=device,
                                         dtype=torch.int32).contiguous()
        if topk.ndim == 4:
            return topk
        if topk.ndim == 3:
            return topk.unsqueeze(2).contiguous()
        if topk.ndim == 2:
            return topk.reshape(int(topk.shape[0]), 1, 1,
                                int(topk.shape[1])).contiguous()
        raise ValueError(
            "selection_topk_indices must be [batch, topk], "
            "[batch, heads, topk], or [batch, heads, q, topk], got "
            f"{tuple(topk.shape)}")

    @staticmethod
    def _require_gs_full_cache_device(tensor: torch.Tensor, *,
                                      name: str,
                                      device: torch.device) -> None:
        if tensor.device.type != device.type:
            raise RuntimeError(
                f"DSA gather-selection requires {name} to be backed by "
                "NPU-accessible swapped memory on the same device family as "
                f"the selection cache. Got {name} on {tensor.device}, "
                f"selection cache on {device}. Check that AscendDSAHotKVStore "
                "allocated DRAM arenas with torch_npu.empty_with_swapped_memory."
            )
        tensor_index = tensor.device.index
        device_index = device.index
        if (tensor_index is not None and device_index is not None
                and int(tensor_index) != int(device_index)):
            raise RuntimeError(
                f"DSA gather-selection requires {name} on the same NPU as "
                f"the selection cache. Got {name} on {tensor.device}, "
                f"selection cache on {device}.")

    @staticmethod
    def _as_device_i32(values, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(values):
            if (values.device == device and values.dtype == torch.int32
                    and values.is_contiguous()):
                return values
            return values.to(device=device, dtype=torch.int32).contiguous()
        return torch.tensor(values, dtype=torch.int32, device=device)

    def gather_selection_update(
        self,
        *,
        selection_topk_indices: torch.Tensor,
        req_pool_entries: torch.Tensor,
        candidate_lens: torch.Tensor,
        selection_block_table: torch.Tensor,
        full_block_table: torch.Tensor,
        nopek_cache_zone: torch.Tensor,
        ropek_cache_zone: torch.Tensor,
        nopek_dram_arena: torch.Tensor,
        ropek_dram_arena: torch.Tensor,
        resident_slot_token_status: torch.Tensor,
        query_start_locs: torch.Tensor,
        query_lens: torch.Tensor,
        query_position_rows: torch.Tensor,
        tail_valid_token_counts: torch.Tensor,
        resident_tail_starts: torch.Tensor,
        budget_lengths: torch.Tensor,
        attention_indices_width: int,
        layer_id: int | None = None,
        existing_attention_indices: torch.Tensor | None = None,
        prebuilt_attention_indices: torch.Tensor | None = None,
        row_modes: torch.Tensor | None = None,
    ) -> DSAGatherSelectionOutput:
        if not torch.is_tensor(selection_topk_indices):
            raise TypeError("selection_topk_indices must be a tensor")
        if layer_id is None:
            raise ValueError("layer_id is required for gather-selection")

        selection_k_rope = self._squeeze_cache_head_dim(
            ropek_cache_zone, "ropek_cache_zone")
        selection_kv_cache = self._squeeze_cache_head_dim(
            nopek_cache_zone, "nopek_cache_zone")
        full_k_rope = self._squeeze_cache_head_dim(
            ropek_dram_arena, "ropek_dram_arena")
        full_kv_cache = self._squeeze_cache_head_dim(
            nopek_dram_arena, "nopek_dram_arena")

        device = selection_kv_cache.device
        self._require_gs_full_cache_device(full_k_rope,
                                           name="full_k_rope",
                                           device=device)
        self._require_gs_full_cache_device(full_kv_cache,
                                           name="full_kv_cache",
                                           device=device)
        topk = self._normalize_selection_topk_indices(
            selection_topk_indices, device=device)
        # resident_slot_token_status is created and shape-checked by
        # DSAResidentTokenPool.  Keep this hot path lean: gather-selection runs
        # once per layer during decode, so repeating full device/dtype/shape
        # validation here only adds host-side overhead.
        status = resident_slot_token_status
        selection_block_table_i32 = self._as_device_i32(selection_block_table,
                                                       device)
        req_pool_entries_i32 = self._as_device_i32(req_pool_entries,
                                                   device).reshape(-1)
        full_block_table_i32 = self._as_device_i32(full_block_table, device)
        candidate_lens_i32 = self._as_device_i32(candidate_lens,
                                                device).reshape(-1)
        if not torch.is_tensor(row_modes):
            raise RuntimeError(
                "DSA gather-selection now requires row_modes metadata; "
                "the legacy compact GS path is disabled.")
        row_modes_i32 = self._as_device_i32(row_modes, device).reshape(-1)
        if int(row_modes_i32.numel()) != int(req_pool_entries_i32.numel()):
            raise RuntimeError(
                "DSA gather-selection row_modes must match full batch: "
                f"row_modes={int(row_modes_i32.numel())}, "
                f"rows={int(req_pool_entries_i32.numel())}")
        debug_enabled = dsa_trace_enabled(
            DSA_TRACE_POINT_GATHER_SELECTION,
            layer_id=layer_id,
        )
        stats_enabled = dsa_trace_enabled(
            DSA_TRACE_POINT_GATHER_SELECTION_STATS,
            layer_id=layer_id,
        )
        debug_sync = debug_enabled and dsa_trace_sync_enabled(
            DSA_TRACE_POINT_GATHER_SELECTION)
        stats_sync = stats_enabled and dsa_trace_sync_enabled(
            DSA_TRACE_POINT_GATHER_SELECTION_STATS)
        if debug_enabled:
            debug_context = {
                "layer_id": int(layer_id),
                "device": str(device),
                "selection_k_rope": _tensor_brief(selection_k_rope),
                "selection_kv_cache": _tensor_brief(selection_kv_cache),
                "selection_block_table":
                _tensor_brief(selection_block_table_i32),
                "status": _tensor_brief(status),
                "req_pool_entries": _tensor_brief(req_pool_entries_i32),
                "topk": _tensor_brief(topk),
                "full_k_rope": _tensor_brief(full_k_rope),
                "full_kv_cache": _tensor_brief(full_kv_cache),
                "full_block_table": _tensor_brief(full_block_table_i32),
                "candidate_lens": _tensor_brief(candidate_lens_i32),
                "row_modes": _tensor_brief(row_modes_i32),
                "status_capacity": int(status.shape[0]),
                "attention_indices_width": int(attention_indices_width),
            }
            logger.info("[DSA GS backend begin] %s", {
                **debug_context,
                "req_pool_entries_sample":
                _tensor_sample(req_pool_entries_i32),
                "candidate_lens_sample":
                _tensor_sample(candidate_lens_i32),
                "row_modes_sample":
                _tensor_sample(row_modes_i32),
                "selection_block_table_sample":
                _tensor_sample(selection_block_table_i32, limit=16),
                "full_block_table_sample":
                _tensor_sample(full_block_table_i32, limit=16),
                "topk_sample":
                _tensor_sample(topk, limit=16),
            })
            if debug_sync:
                _dsa_gs_backend_sync("before_gs_op", layer_id, debug_context)
        if stats_enabled:
            stats_context = _build_gather_selection_overlap_stats(
                layer_id=int(layer_id),
                topk=topk,
                status=status,
                req_pool_entries=req_pool_entries_i32,
                row_modes=row_modes_i32,
                candidate_lens=candidate_lens_i32,
            )
            logger.info("[DSA GS overlap stats] %s", stats_context)
            if stats_sync:
                _dsa_gs_backend_sync("after_gs_overlap_stats", layer_id,
                                     stats_context)
        if torch.is_tensor(prebuilt_attention_indices):
            attention_indices = prebuilt_attention_indices
            if attention_indices.device != device:
                raise RuntimeError(
                    "prebuilt DSA attention indices must stay on the "
                    f"selection cache device: {attention_indices.device} "
                    f"vs {device}")
            if attention_indices.dtype != torch.int32:
                raise RuntimeError(
                    "prebuilt DSA attention indices must be int32, got "
                    f"{attention_indices.dtype}")
        else:
            attention_indices = torch.empty(
                (int(req_pool_entries_i32.numel()),
                 int(attention_indices_width)),
                dtype=torch.int32,
                device=device,
            )
        # Gather-selection has two independent metadata effects:
        #
        # 1. resident_slot_token_status is updated in place by the kernel.
        #    The lower-level op schema still calls this tensor
        #    selection_kv_block_status for compatibility with the original GS
        #    operator.  Semantically it stores the persistent mapping
        #    resident_slot -> original token/segment id, and is used by the
        #    next decode step to detect HBM hits.
        # 2. attention_indices is the SFA-facing index tensor for this forward.
        #    For sparse rows those values are resident logical slot ids
        #    (0..budget-1 plus resident tail slots), not original token ids.
        #    Do not sort or interpret them as sequence positions.
        torch.ops._C_ascend.gather_selection_kv_cache(
            selection_k_rope,
            selection_kv_cache,
            selection_block_table_i32,
            status,
            req_pool_entries_i32,
            topk,
            full_k_rope,
            full_kv_cache,
            full_block_table_i32,
            candidate_lens_i32,
            row_modes_i32,
            self._as_device_i32(budget_lengths, device).reshape(-1),
            self._as_device_i32(tail_valid_token_counts,
                                device).reshape(-1),
            self._as_device_i32(resident_tail_starts, device).reshape(-1),
            self._as_device_i32(query_position_rows, device),
            attention_indices,
        )
        if debug_enabled:
            if debug_sync:
                _dsa_gs_backend_sync("after_gs_op", layer_id, debug_context)
            logger.info("[DSA GS backend op done] %s", debug_context)
        if debug_enabled:
            attention_context = {
                **debug_context,
                "attention_indices": _tensor_brief(attention_indices),
                "attention_indices_sample":
                _tensor_sample(attention_indices, limit=16),
            }
            if debug_sync:
                _dsa_gs_backend_sync("after_attention_indices", layer_id,
                                     attention_context)
            logger.info("[DSA GS backend attention indices] %s",
                        attention_context)
        return DSAGatherSelectionOutput(
            attention_indices=attention_indices,
            resident_update_committed=True,
        )
