"""DSA HBM resident sparse budget 资源池。

本文件维护每个请求在 HBM resident plane 上的固定预算槽位，包括请求到
resident row 的映射、layer 级 resident count 张量视图、GS resident status
张量，以及图模式可复用的资源池。它只管理 HBM resident 资源的 Python 侧
生命周期，不负责 DRAM 热层块分配、scheduler 侧 block admission，也不负责
SFA attention_indices 的构造。

当前 row-mode gather-selection 路径中，resident slot -> 原始 token/segment
的权威映射由 resident_slot_token_status 持有并被 GS kernel 原址刷新。底层
算子接口仍沿用历史入参名 selection_kv_block_status；Python/DSA 层使用更
贴近语义的 resident_slot_token_status，表示“resident 槽里当前是哪一个
原始 token/segment”。
"""

from __future__ import annotations

from typing import Hashable, NamedTuple

import torch


class DSAResidentLayerResourceView(NamedTuple):
    counts: torch.Tensor
    pool_indices: torch.Tensor
    device: torch.device
    max_tokens: int


class DSAResidentTokenPool:
    """Per-worker resident request metadata.

    Besides pool ownership and resident counts, this pool owns the fixed 5-D
    resident_slot_token_status tensor used by gather-selection.  The tensor is
    the authoritative resident slot -> original token/segment mapping and is
    cleared together with the resident pool row.
    """

    def __init__(
        self,
        max_reqs: int,
        num_layers: int,
        max_tokens: int,
        *,
        device: torch.device | str | None = None,
    ):
        if max_reqs <= 0:
            raise ValueError(f"max_reqs must be positive, got {max_reqs}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        self.max_reqs = int(max_reqs)
        self.num_layers = int(num_layers)
        self.max_tokens = int(max_tokens)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self._free_indices = list(range(self.max_reqs))
        self._request_to_index: dict[Hashable, int] = {}
        self._cached_counts = torch.zeros(
            (self.max_reqs, self.num_layers),
            dtype=torch.int32,
            device=self.device,
        )
        # 当前 DSA row-mode GS 路径的 topK 等于固定 resident budget，因此
        # resident slot 状态可以资源池化成一张稳定 5D tensor：
        # [layer, pool_idx, 1, 1, resident_slot]。逐层传给底层 GS op 时只
        # 取 self._resident_slot_token_status[layer_id] 这个 4D view，避免
        # 运行时按 layer lazy 分配，也让 request release/preempt 能一次
        # 清掉所有 layer 的同一 pool row。
        self._resident_slot_token_status = torch.full(
            (self.num_layers, self.max_reqs, 1, 1, self.max_tokens + 1),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

    @property
    def req_hbm_cached_token_counts(self) -> torch.Tensor:
        return self._cached_counts

    def acquire(self, request_id: Hashable) -> int:
        current = self._request_to_index.get(request_id)
        if current is not None:
            return current
        if not self._free_indices:
            raise RuntimeError(
                "No free DSA resident metadata slot is available")

        pool_idx = self._free_indices.pop(0)
        self._request_to_index[request_id] = pool_idx
        self._clear_index(pool_idx)
        return pool_idx

    def release(self, request_id: Hashable) -> None:
        pool_idx = self._request_to_index.pop(request_id, None)
        if pool_idx is None:
            return
        self._clear_index(pool_idx)
        self._free_indices.insert(0, pool_idx)

    def get_index(self, request_id: Hashable) -> int | None:
        return self._request_to_index.get(request_id)

    def clear_request(self, request_id: Hashable) -> None:
        pool_idx = self._require_index(request_id)
        self._clear_index(pool_idx)

    def get_layer_resource_view_by_index(
        self,
        pool_indices,
        layer_id: int,
    ) -> DSAResidentLayerResourceView:
        layer_id = self._normalize_layer_id(layer_id)
        if torch.is_tensor(pool_indices):
            pool_indices_tensor = pool_indices.to(
                device=self._cached_counts.device,
                dtype=torch.int32,
            )
        else:
            pool_indices_tensor = torch.as_tensor(
                [int(pool_idx) for pool_idx in pool_indices],
                dtype=torch.int32,
                device=self._cached_counts.device,
            )
        return DSAResidentLayerResourceView(
            counts=self._cached_counts[:, layer_id],
            pool_indices=pool_indices_tensor,
            device=self._cached_counts.device,
            max_tokens=self.max_tokens,
        )

    def get_resident_slot_token_status(self, *, layer_id: int,
                                       topk: int) -> torch.Tensor:
        """Return the per-layer resident slot mapping used by GS.

        resident_slot_token_status 是当前 row-mode GS 路径的权威映射：
        status[pool_idx, 0, 0, resident_slot] = original token/segment id。
        底层算子接口因为历史原因仍叫 selection_kv_block_status，但它的值
        语义不是 block id，而是 resident sparse budget slot 当前承载的原始
        token/segment id。它由 GS kernel 原址刷新，下一轮 decode 继续用同
        一张表判断 hit/miss。因此这份状态必须跟 resident pool 的 request
        行生命周期绑定，而不是藏在无状态算子 backend 里临时创建。
        """
        layer_id = self._normalize_layer_id(layer_id)
        topk = int(topk)
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        if topk > self.max_tokens:
            raise ValueError(
                f"topk {topk} exceeds resident budget {self.max_tokens}")
        if topk != self.max_tokens:
            raise ValueError(
                "resident_slot_token_status uses fixed resident budget "
                f"topk={self.max_tokens}, got {topk}")
        return self._resident_slot_token_status[layer_id]

    def ensure_resident_slot_token_statuses(
            self, *, topk: int | None = None) -> None:
        """Pre-create resident slot token status tensors for every layer.

        图模式 capture/replay 希望关键张量地址提前稳定下来；eager 模式也不应
        在每层首次 after_indexer 时夹杂一次隐式分配。当前 status 已在
        __init__ 中固定创建，这里保留为初始化流程里的语义检查入口。
        """
        topk = self.max_tokens if topk is None else int(topk)
        if topk != self.max_tokens:
            raise ValueError(
                "resident_slot_token_status uses fixed resident budget "
                f"topk={self.max_tokens}, got {topk}")

    def clear_resident_slot_token_status(self, pool_idx: int) -> None:
        pool_idx = int(pool_idx)
        if pool_idx < 0 or pool_idx >= self.max_reqs:
            return
        self._resident_slot_token_status[:, pool_idx].fill_(-1)

    def clear_resident_slot_token_status_prefix(self, row_count: int) -> None:
        """Clear status rows ``[0, row_count)`` for every layer at once.

        图 capture dummy batch 使用连续的 pool row 前缀。固定 5D status 池
        允许把原先逐 row 的 Python 循环合成一次 strided tensor fill，减少
        capture/restore 阶段的 host 调度和 kernel launch 数量。
        """
        row_count = min(max(int(row_count), 0), self.max_reqs)
        if row_count == 0:
            return
        self._resident_slot_token_status[:, :row_count].fill_(-1)

    def _clear_index(self, pool_idx: int) -> None:
        self._cached_counts[pool_idx].zero_()
        self.clear_resident_slot_token_status(pool_idx)

    def _require_index(self, request_id: Hashable) -> int:
        pool_idx = self._request_to_index.get(request_id)
        if pool_idx is None:
            raise KeyError(f"DSA request {request_id!r} has no resident slot")
        return pool_idx

    def _normalize_layer_id(self, layer_id: int) -> int:
        layer_id = int(layer_id)
        if layer_id < 0 or layer_id >= self.num_layers:
            raise IndexError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})")
        return layer_id
