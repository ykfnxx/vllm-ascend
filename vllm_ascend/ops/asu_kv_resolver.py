#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import torch

INVALID = -1
ASU_ONLY = 0
HBM_RESIDENT = 1
TAIL_HBM = 2


def _scalar_tensor_value(tensor: torch.Tensor) -> int:
    return int(tensor.reshape(-1)[0].item())


def _set_scalar_tensor_value(tensor: torch.Tensor, value: int) -> None:
    tensor.reshape(-1)[0] = value


def _tensor_value(tensor: torch.Tensor, index: int) -> int:
    return int(tensor.reshape(-1)[index].item())


def _set_tensor_value(tensor: torch.Tensor, index: int, value: int) -> None:
    tensor.reshape(-1)[index] = value


def _check_slot_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() < 1:
        raise ValueError(f"{name} must have a slot dimension")


def _check_kv_pair_shapes(
    original_kv_cache_0: torch.Tensor,
    original_kv_cache_1: torch.Tensor,
    managed_kv_cache_0: torch.Tensor,
    managed_kv_cache_1: torch.Tensor,
) -> None:
    _check_slot_tensor("original_kv_cache_0", original_kv_cache_0)
    _check_slot_tensor("original_kv_cache_1", original_kv_cache_1)
    _check_slot_tensor("managed_kv_cache_0", managed_kv_cache_0)
    _check_slot_tensor("managed_kv_cache_1", managed_kv_cache_1)
    if original_kv_cache_0.shape[1:] != managed_kv_cache_0.shape[1:]:
        raise ValueError("original_kv_cache_0 and managed_kv_cache_0 must "
                         "have matching per-slot shapes")
    if original_kv_cache_1.shape[1:] != managed_kv_cache_1.shape[1:]:
        raise ValueError("original_kv_cache_1 and managed_kv_cache_1 must "
                         "have matching per-slot shapes")


def _pop_free_slot(free_slot_stack: torch.Tensor,
                   free_slot_count: torch.Tensor) -> int:
    count = _scalar_tensor_value(free_slot_count)
    if count <= 0:
        raise RuntimeError("free_slot_stack does not have enough slots")

    next_count = count - 1
    slot = _tensor_value(free_slot_stack, next_count)
    _set_scalar_tensor_value(free_slot_count, next_count)
    return slot


def _resolve_tail_source_slot(original_block_table: torch.Tensor,
                              token_id: int,
                              block_size: int) -> int:
    logical_block = token_id // block_size
    offset = token_id % block_size
    block_table = original_block_table.reshape(-1)
    if logical_block >= block_table.numel():
        raise IndexError("original_block_table does not cover token_id "
                         f"{token_id}")

    physical_block = _tensor_value(block_table, logical_block)
    return physical_block * block_size + offset


def _copy_full_kv_pair(
    original_kv_cache_0: torch.Tensor,
    original_kv_cache_1: torch.Tensor,
    source_slot: int,
    managed_kv_cache_0: torch.Tensor,
    managed_kv_cache_1: torch.Tensor,
    managed_slot: int,
) -> None:
    if source_slot < 0 or source_slot >= original_kv_cache_0.size(0):
        raise IndexError(f"source_slot {source_slot} is out of range")
    if source_slot >= original_kv_cache_1.size(0):
        raise IndexError(f"source_slot {source_slot} is out of range")
    if managed_slot < 0 or managed_slot >= managed_kv_cache_0.size(0):
        raise IndexError(f"managed_slot {managed_slot} is out of range")
    if managed_slot >= managed_kv_cache_1.size(0):
        raise IndexError(f"managed_slot {managed_slot} is out of range")

    managed_kv_cache_0[managed_slot].copy_(original_kv_cache_0[source_slot])
    managed_kv_cache_1[managed_slot].copy_(original_kv_cache_1[source_slot])


def _install_token_to_managed_cache(
    token_id: int,
    source_slot: int,
    token_state: torch.Tensor,
    hbm_slot_of_token: torch.Tensor,
    slot_owner_token: torch.Tensor,
    free_slot_stack: torch.Tensor,
    free_slot_count: torch.Tensor,
    original_kv_cache_0: torch.Tensor,
    original_kv_cache_1: torch.Tensor,
    managed_kv_cache_0: torch.Tensor,
    managed_kv_cache_1: torch.Tensor,
) -> int:
    slot = _pop_free_slot(free_slot_stack, free_slot_count)
    _copy_full_kv_pair(
        original_kv_cache_0,
        original_kv_cache_1,
        source_slot,
        managed_kv_cache_0,
        managed_kv_cache_1,
        slot,
    )
    _set_tensor_value(token_state, token_id, HBM_RESIDENT)
    _set_tensor_value(hbm_slot_of_token, token_id, slot)
    _set_tensor_value(slot_owner_token, slot, token_id)
    return slot


def asu_resolve_kv_slots_single_req(
    original_topk_indices: torch.Tensor,
    actual_seq_len: int,
    managed_prefix_len: int,
    token_state: torch.Tensor,
    asu_record_addr: torch.Tensor,
    hbm_slot_of_token: torch.Tensor,
    slot_owner_token: torch.Tensor,
    free_slot_stack: torch.Tensor,
    free_slot_count: torch.Tensor,
    original_block_table: torch.Tensor,
    original_kv_cache_0: torch.Tensor,
    original_kv_cache_1: torch.Tensor,
    managed_kv_cache_0: torch.Tensor,
    managed_kv_cache_1: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    del managed_prefix_len
    if actual_seq_len < 0:
        raise ValueError("actual_seq_len must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    _check_kv_pair_shapes(
        original_kv_cache_0,
        original_kv_cache_1,
        managed_kv_cache_0,
        managed_kv_cache_1,
    )
    resolved_kv_slots = torch.empty(
        original_topk_indices.shape,
        device=original_topk_indices.device,
        dtype=torch.int32,
    )

    topk_flat = original_topk_indices.reshape(-1)
    resolved_flat = resolved_kv_slots.reshape(-1)
    for index in range(topk_flat.numel()):
        token_id = _tensor_value(topk_flat, index)
        if token_id < 0 or token_id >= actual_seq_len:
            raise IndexError(f"token_id {token_id} is out of range")

        state = _tensor_value(token_state, token_id)
        if state == HBM_RESIDENT:
            slot = _tensor_value(hbm_slot_of_token, token_id)
        elif state == ASU_ONLY:
            source_slot = _tensor_value(asu_record_addr, token_id)
            slot = _install_token_to_managed_cache(
                token_id,
                source_slot,
                token_state,
                hbm_slot_of_token,
                slot_owner_token,
                free_slot_stack,
                free_slot_count,
                original_kv_cache_0,
                original_kv_cache_1,
                managed_kv_cache_0,
                managed_kv_cache_1,
            )
        elif state == TAIL_HBM:
            source_slot = _resolve_tail_source_slot(
                original_block_table,
                token_id,
                block_size,
            )
            slot = _install_token_to_managed_cache(
                token_id,
                source_slot,
                token_state,
                hbm_slot_of_token,
                slot_owner_token,
                free_slot_stack,
                free_slot_count,
                original_kv_cache_0,
                original_kv_cache_1,
                managed_kv_cache_0,
                managed_kv_cache_1,
            )
        else:
            raise ValueError(f"unsupported token_state {state} for token_id "
                             f"{token_id}")

        _set_tensor_value(resolved_flat, index, slot)

    return resolved_kv_slots
