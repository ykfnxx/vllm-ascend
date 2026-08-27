# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from vllm_ascend.dsa_offload.hot_cache import (
    HotCacheLayout,
    fixed_memory_bytes,
    resize_target_tensors,
    validate_target_tensors,
)


@dataclass
class TensorConfig:
    shared_by: list[str]
    size: int


def test_d_only_and_mixed_layout() -> None:
    d_only = HotCacheLayout(128, 2, 16)
    mixed = HotCacheLayout(128, 2, 16, hot_block_base=100)

    assert d_only.resident_blocks == 64
    assert d_only.replaceable_blocks == 16
    assert d_only.transient_blocks == 1
    assert d_only.hot_blocks_per_row == 82
    assert d_only.tail_base == 10240
    assert d_only.fallback_slot == 10368
    assert d_only.staging_base == 10369
    assert d_only.row_stride == 10496
    assert mixed.row_block_base(0) == 100
    assert mixed.row_block_base(1) == 182
    assert mixed.block_table(torch.tensor([0, 1], dtype=torch.int32))[0, :3].tolist() == [100, 101, 102]


def test_target_tensor_resize_keeps_hot_blocks_outside_regular_pool() -> None:
    spec = SimpleNamespace(page_size_bytes=512)
    target_specs = {"model.layers.0.attn": spec}
    tensor = TensorConfig(["model.layers.0.attn"], 100 * 512)
    kv_config = SimpleNamespace(num_blocks=100, kv_cache_tensors=[tensor])

    d_only = HotCacheLayout(128, 1, 1)
    resize_target_tensors(kv_config, target_specs, d_only, "kv_consumer")
    assert tensor.size == d_only.hot_blocks * 512
    validate_target_tensors(kv_config, target_specs, d_only, "kv_consumer")

    mixed = HotCacheLayout(128, 1, 1, hot_block_base=kv_config.num_blocks)
    resize_target_tensors(kv_config, target_specs, mixed, "kv_both")
    assert tensor.size == (kv_config.num_blocks + mixed.hot_blocks) * 512
    assert mixed.row_block_base(0) >= kv_config.num_blocks
    validate_target_tensors(kv_config, target_specs, mixed, "kv_both")


def test_fixed_memory_includes_payload_and_one_state_per_cohort() -> None:
    layout = HotCacheLayout(128, 2, 16)
    specs = [SimpleNamespace(page_size_bytes=64), SimpleNamespace(page_size_bytes=96)]

    payload = layout.hot_blocks * 160
    lookup = 2 * (128 * 1024 + 10 * 1024 + 2 * 1024 + 16) * 4
    assert fixed_memory_bytes(layout, specs, 3) == payload + 3 * lookup
