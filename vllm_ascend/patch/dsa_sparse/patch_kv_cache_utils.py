"""Build independent Indexer and MLA KV cache planes on vLLM v0.18."""

from collections import defaultdict

from vllm.utils.math_utils import cdiv
from vllm.v1.core import kv_cache_utils as kv_utils
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheSpec

from vllm_ascend.dsa_sparse.dsa_config import attach_dsa_sparse_cache_attrs
from vllm_ascend.dsa_sparse.dsa_spec_utils import (
    is_dsa_indexer_spec,
    is_dsa_mla_resident_spec,
)

_original_get_kv_cache_config_from_groups = (
    kv_utils.get_kv_cache_config_from_groups
)
_original_get_kv_cache_groups = kv_utils.get_kv_cache_groups
_original_get_kv_cache_configs = kv_utils.get_kv_cache_configs
_original_max_memory_usage_bytes_from_groups = (
    kv_utils._max_memory_usage_bytes_from_groups
)


def _has_indexer_spec(kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
    return any(is_dsa_indexer_spec(spec) for spec in kv_cache_specs.values())


def _has_indexer_group(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    return any(
        is_dsa_indexer_spec(group.kv_cache_spec)
        for group in kv_cache_groups
    )


def _get_group_num_blocks(
    vllm_config,
    available_memory: int,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[int, list[int]]:
    ratio = int(vllm_config.cache_config.dsa_indexer_mla_block_ratio)
    assert ratio > 0
    weighted_page_size = sum(
        group.kv_cache_spec.page_size_bytes
        * len(group.layer_names)
        * (ratio if is_dsa_indexer_spec(group.kv_cache_spec) else 1)
        for group in kv_cache_groups
    )
    base_num_blocks = available_memory // weighted_page_size
    base_num_blocks = base_num_blocks // 128 * 128
    base_num_blocks = kv_utils.may_override_num_blocks(
        vllm_config, base_num_blocks
    )
    assert base_num_blocks > 0
    group_num_blocks = [
        base_num_blocks * ratio
        if is_dsa_indexer_spec(group.kv_cache_spec)
        else base_num_blocks
        for group in kv_cache_groups
    ]
    return base_num_blocks, group_num_blocks


def _get_kv_cache_groups_by_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    same_spec_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    block_size = next(iter(kv_cache_specs.values())).block_size
    for layer_name, layer_spec in kv_cache_specs.items():
        assert layer_spec.block_size == block_size
        same_spec_layers[layer_spec].append(layer_name)
    return kv_utils.create_kv_cache_group_specs(
        kv_cache_specs, list(same_spec_layers.values())
    )


def _get_kv_cache_groups(
    vllm_config,
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    if not _has_indexer_spec(kv_cache_specs):
        return _original_get_kv_cache_groups(vllm_config, kv_cache_specs)
    return _get_kv_cache_groups_by_spec(kv_cache_specs)


def _get_kv_cache_config_from_groups(
    vllm_config,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    if not _has_indexer_group(kv_cache_groups):
        return _original_get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )

    base_num_blocks, group_num_blocks = _get_group_num_blocks(
        vllm_config, available_memory, kv_cache_groups
    )
    kv_cache_tensors = []
    for group, num_blocks in zip(kv_cache_groups, group_num_blocks):
        group.dsa_num_blocks = num_blocks
        kv_cache_tensors.extend(
            kv_utils.KVCacheTensor(
                size=group.kv_cache_spec.page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in group.layer_names
        )
    return KVCacheConfig(
        num_blocks=base_num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def _max_memory_usage_bytes_from_groups(
    vllm_config,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    if not _has_indexer_group(kv_cache_groups):
        return _original_max_memory_usage_bytes_from_groups(
            vllm_config, kv_cache_groups
        )

    sparse_budget = int(vllm_config.cache_config.dsa_hbm_sparse_budget)
    total_bytes = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if is_dsa_indexer_spec(spec):
            group_bytes = spec.max_memory_usage_bytes(vllm_config)
        else:
            assert is_dsa_mla_resident_spec(spec)
            resident_blocks = cdiv(
                sparse_budget + spec.block_size, spec.block_size
            )
            group_bytes = resident_blocks * spec.page_size_bytes
        total_bytes += len(group.layer_names) * group_bytes
    return total_bytes


def _normalize_group_num_blocks(
    kv_cache_configs: list[KVCacheConfig],
    vllm_config,
) -> None:
    if not any(
        _has_indexer_group(config.kv_cache_groups)
        for config in kv_cache_configs
    ):
        return

    base_num_blocks = min(config.num_blocks for config in kv_cache_configs)
    ratio = int(vllm_config.cache_config.dsa_indexer_mla_block_ratio)
    for config in kv_cache_configs:
        for group in config.kv_cache_groups:
            group.dsa_num_blocks = (
                base_num_blocks * ratio
                if is_dsa_indexer_spec(group.kv_cache_spec)
                else base_num_blocks
            )
        for tensor in config.kv_cache_tensors:
            group = next(
                group
                for group in config.kv_cache_groups
                if tensor.shared_by[0] in group.layer_names
            )
            tensor.size = (
                group.kv_cache_spec.page_size_bytes * group.dsa_num_blocks
            )


def _get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
    attach_dsa_sparse_cache_attrs(vllm_config)
    configs = _original_get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_memory
    )
    _normalize_group_num_blocks(configs, vllm_config)
    return configs


def install_dsa_kv_cache_utils_patch() -> None:
    kv_utils.get_kv_cache_groups = _get_kv_cache_groups
    kv_utils.get_kv_cache_config_from_groups = _get_kv_cache_config_from_groups
    kv_utils._max_memory_usage_bytes_from_groups = (
        _max_memory_usage_bytes_from_groups
    )
    kv_utils.get_kv_cache_configs = _get_kv_cache_configs


install_dsa_kv_cache_utils_patch()
