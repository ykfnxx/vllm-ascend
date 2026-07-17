"""Persistent lookup-index state for DSA HBM resident KV slots."""

from __future__ import annotations

from typing import Hashable, NamedTuple

import torch

DSA_LOOKUP_INDEX_CAPACITY = 128 * 1024
DSA_LOOKUP_RESIDENT_TOKENS = 8 * 1024
DSA_LOOKUP_QUERY_TOKENS = 2 * 1024
DSA_LOOKUP_TOTAL_SLOTS = 10 * 1024
DSA_FREE_HEAD_STRIDE = 16


class DSAResidentLayerResourceView(NamedTuple):
    counts: torch.Tensor
    pool_indices: torch.Tensor
    device: torch.device
    max_tokens: int


class DSAResidentLookupState(NamedTuple):
    """One layer's persistent lookup state across all request-pool rows."""

    token_to_slot: torch.Tensor
    slot_to_token: torch.Tensor
    free_slots: torch.Tensor
    free_head: torch.Tensor


class DSAResidentTokenPool:
    """Own request rows and per-layer token-to-slot lookup state.

    ``resident_tokens`` is the number of occupied slots kept after maintenance.
    ``free_slot_tokens`` is the lookup headroom reserved for one decode step.
    The physical resident address space is their sum. Lookup allocates misses
    from the headroom and maintenance evicts the same number of non-protected
    historical entries, restoring the headroom before the next decode step.
    """

    def __init__(
        self,
        max_reqs: int,
        num_layers: int,
        index_capacity: int,
        resident_tokens: int,
        free_slot_tokens: int,
        *,
        device: torch.device | str | None = None,
    ):
        if max_reqs <= 0:
            raise ValueError(f"max_reqs must be positive, got {max_reqs}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if index_capacity <= 0:
            raise ValueError(
                f"index_capacity must be positive, got {index_capacity}")
        if resident_tokens <= free_slot_tokens:
            raise ValueError(
                "resident_tokens must be greater than free_slot_tokens: "
                f"resident_tokens={resident_tokens}, "
                f"free_slot_tokens={free_slot_tokens}")
        if free_slot_tokens <= 0:
            raise ValueError(
                f"free_slot_tokens must be positive, got {free_slot_tokens}")

        self.max_reqs = int(max_reqs)
        self.num_layers = int(num_layers)
        self.index_capacity = int(index_capacity)
        self.resident_tokens = int(resident_tokens)
        self.free_slot_tokens = int(free_slot_tokens)
        self.total_slots = self.resident_tokens + self.free_slot_tokens
        # Kept for the existing graph-admission/resource-view API. It now means
        # stable occupied lookup entries, not the current TopK width.
        self.max_tokens = self.resident_tokens
        self.device = (torch.device("cpu") if device is None else
                       torch.device(device))
        self._free_indices = list(range(self.max_reqs))
        self._request_to_index: dict[Hashable, int] = {}
        self._cached_counts = torch.zeros(
            (self.max_reqs, self.num_layers),
            dtype=torch.int32,
            device=self.device,
        )
        self._token_to_slot = torch.full(
            (self.num_layers, self.max_reqs, self.index_capacity),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._slot_to_token = torch.full(
            (self.num_layers, self.max_reqs, self.total_slots),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._free_slot_template = torch.arange(
            self.resident_tokens,
            self.total_slots,
            dtype=torch.int32,
            device=self.device,
        )
        self._free_slots = self._free_slot_template.view(1, 1, -1).expand(
            self.num_layers, self.max_reqs, -1).clone()
        # Isolate each scalar head in its own 64-byte cache line so lookup can
        # update different requests from different AIV cores.
        self._free_head = torch.zeros(
            (self.num_layers, self.max_reqs, DSA_FREE_HEAD_STRIDE),
            dtype=torch.int32,
            device=self.device,
        )

    @property
    def req_hbm_cached_token_counts(self) -> torch.Tensor:
        return self._cached_counts

    @property
    def allocated_tensor_bytes(self) -> int:
        """Return the storage bytes owned by the persistent lookup state."""
        tensors = (
            self._cached_counts,
            self._token_to_slot,
            self._slot_to_token,
            self._free_slot_template,
            self._free_slots,
            self._free_head,
        )
        return sum(tensor.numel() * tensor.element_size()
                   for tensor in tensors)

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
        self._clear_index(self._require_index(request_id))

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
            max_tokens=self.resident_tokens,
        )

    def get_layer_lookup_state(self, layer_id: int) -> DSAResidentLookupState:
        layer_id = self._normalize_layer_id(layer_id)
        return DSAResidentLookupState(
            token_to_slot=self._token_to_slot[layer_id],
            slot_to_token=self._slot_to_token[layer_id],
            free_slots=self._free_slots[layer_id],
            free_head=self._free_head[layer_id],
        )

    def clear_lookup_state_prefix(self, row_count: int) -> None:
        row_count = min(max(int(row_count), 0), self.max_reqs)
        if row_count == 0:
            return
        self._token_to_slot[:, :row_count].fill_(-1)
        self._slot_to_token[:, :row_count].fill_(-1)
        self._reset_free_slots(slice(0, row_count))
        self._free_head[:, :row_count].zero_()

    def _reset_free_slots(self, pool_rows) -> None:
        self._free_slots[:, pool_rows].copy_(
            self._free_slot_template.view(1, 1, -1).expand(
                self.num_layers,
                self._free_slots[:, pool_rows].shape[1],
                -1,
            ))

    def _clear_index(self, pool_idx: int) -> None:
        self._cached_counts[pool_idx].zero_()
        self._token_to_slot[:, pool_idx].fill_(-1)
        self._slot_to_token[:, pool_idx].fill_(-1)
        self._free_slots[:, pool_idx].copy_(
            self._free_slot_template.view(1, -1).expand(
                self.num_layers, -1))
        self._free_head[:, pool_idx].zero_()

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
