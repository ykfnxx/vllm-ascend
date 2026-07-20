"""Build independent Indexer and MLA KV cache planes on vLLM v0.18."""

from collections import defaultdict
from functools import partial

from vllm.logger import init_logger
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

logger = init_logger(__name__)


def _has_indexer_spec(kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
    return any(is_dsa_indexer_spec(spec) for spec in kv_cache_specs.values())


def _has_indexer_group(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    return any(
        is_dsa_indexer_spec(group.kv_cache_spec)
        for group in kv_cache_groups
    )


def _get_local_max_model_len(vllm_config) -> int:
    cp_world_size = (
        int(vllm_config.parallel_config.decode_context_parallel_size)
        * int(vllm_config.parallel_config.prefill_context_parallel_size)
    )
    return cdiv(
        int(vllm_config.model_config.max_model_len), cp_world_size)


def _get_sparse_mla_num_blocks(vllm_config) -> int:
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        return int(override)

    block_size = int(vllm_config.cache_config.block_size)
    max_model_len = _get_local_max_model_len(vllm_config)
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    resident_tokens = int(
        vllm_config.cache_config.dsa_hbm_resident_tokens)
    sparse_topk = int(vllm_config.cache_config.dsa_hbm_sparse_budget)

    dense_blocks = cdiv(max_model_len, block_size)
    resident_blocks = cdiv(
        resident_tokens + sparse_topk + block_size, block_size)
    global_dense_blocks = cdiv(
        int(vllm_config.model_config.max_model_len), block_size)
    secondary_request_blocks = (
        dense_blocks
        if global_dense_blocks <= resident_blocks
        else resident_blocks
    )
    # One pool block is permanently reserved as vLLM's null block. The usable
    # MLA capacity holds one maximum-length dense request while every other
    # active request keeps only its sparse resident window.
    return (
        dense_blocks
        + max(0, max_num_seqs - 1) * secondary_request_blocks
        + 1
    )


def _get_group_num_blocks(
    vllm_config,
    available_memory: int,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[int, list[int]]:
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        num_blocks = int(override)
        return num_blocks, [num_blocks for _ in kv_cache_groups]

    mla_num_blocks = _get_sparse_mla_num_blocks(vllm_config)
    mla_memory = sum(
        group.kv_cache_spec.page_size_bytes
        * len(group.layer_names)
        * mla_num_blocks
        for group in kv_cache_groups
        if is_dsa_mla_resident_spec(group.kv_cache_spec)
    )
    indexer_bytes_per_block = sum(
        group.kv_cache_spec.page_size_bytes * len(group.layer_names)
        for group in kv_cache_groups
        if is_dsa_indexer_spec(group.kv_cache_spec)
    )
    assert indexer_bytes_per_block > 0
    remaining_memory = available_memory - mla_memory
    indexer_num_blocks = remaining_memory // indexer_bytes_per_block

    block_size = int(vllm_config.cache_config.block_size)
    min_indexer_blocks = cdiv(
        _get_local_max_model_len(vllm_config), block_size) + 1
    if indexer_num_blocks < min_indexer_blocks:
        required_memory = (
            mla_memory + min_indexer_blocks * indexer_bytes_per_block)
        raise ValueError(
            "No available memory for the DSA cache layout: "
            f"available_memory={available_memory}, "
            f"required_memory={required_memory}, "
            f"mla_num_blocks={mla_num_blocks}, "
            f"min_indexer_blocks={min_indexer_blocks}")

    group_num_blocks = [
        indexer_num_blocks
        if is_dsa_indexer_spec(group.kv_cache_spec)
        else mla_num_blocks
        for group in kv_cache_groups
    ]
    return indexer_num_blocks, group_num_blocks


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

    indexer_num_blocks, group_num_blocks = _get_group_num_blocks(
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
        num_blocks=indexer_num_blocks,
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

    total_bytes = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if is_dsa_indexer_spec(spec):
            # The dense Indexer plane must hold one maximum-length request and
            # its block pool's permanent null block.
            group_bytes = (
                spec.max_memory_usage_bytes(vllm_config)
                + spec.page_size_bytes
            )
        else:
            assert is_dsa_mla_resident_spec(spec)
            group_bytes = (
                _get_sparse_mla_num_blocks(vllm_config)
                * spec.page_size_bytes
            )
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

    indexer_num_blocks = min(
        config.num_blocks for config in kv_cache_configs)
    mla_num_blocks = _get_sparse_mla_num_blocks(vllm_config)
    for config in kv_cache_configs:
        config.num_blocks = indexer_num_blocks
        for group in config.kv_cache_groups:
            group.dsa_num_blocks = (
                indexer_num_blocks
                if is_dsa_indexer_spec(group.kv_cache_spec)
                else mla_num_blocks
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

    block_size = int(vllm_config.cache_config.block_size)
    max_mla_memory = max(
        sum(
            group.kv_cache_spec.page_size_bytes
            * len(group.layer_names)
            * mla_num_blocks
            for group in config.kv_cache_groups
            if is_dsa_mla_resident_spec(group.kv_cache_spec)
        )
        for config in kv_cache_configs
    )
    max_indexer_memory = max(
        sum(
            group.kv_cache_spec.page_size_bytes
            * len(group.layer_names)
            * indexer_num_blocks
            for group in config.kv_cache_groups
            if is_dsa_indexer_spec(group.kv_cache_spec)
        )
        for config in kv_cache_configs
    )
    logger.info_once(
        "DSA KV cache layout: MLA blocks=%d, Indexer blocks=%d, "
        "max MLA memory per worker=%.2f GiB, "
        "max Indexer memory per worker=%.2f GiB, "
        "Indexer capacity=%d tokens",
        mla_num_blocks,
        indexer_num_blocks,
        max_mla_memory / (1024**3),
        max_indexer_memory / (1024**3),
        max(0, indexer_num_blocks - 1) * block_size,
        scope="local",
    )
    cp_world_size = (
        int(vllm_config.parallel_config.decode_context_parallel_size)
        * int(vllm_config.parallel_config.prefill_context_parallel_size)
    )
    indexer_capacity_tokens = (
        max(0, indexer_num_blocks - 1) * block_size * cp_world_size)
    dense_blocks_per_worker = cdiv(
        _get_local_max_model_len(vllm_config), block_size)
    max_concurrency = min(
        int(vllm_config.scheduler_config.max_num_seqs),
        max(0, indexer_num_blocks - 1) / dense_blocks_per_worker,
    )
    logger.info_once(
        "GPU KV cache size: %s tokens",
        f"{indexer_capacity_tokens:,}",
        scope="local",
    )
    logger.info_once(
        "Maximum DSA sparse concurrency for %s tokens per request: %.2fx",
        f"{int(vllm_config.model_config.max_model_len):,}",
        max_concurrency,
        scope="local",
    )


def _get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
    attach_dsa_sparse_cache_attrs(vllm_config)
    if not any(_has_indexer_spec(specs) for specs in kv_cache_specs):
        return _original_get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_memory)

    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for worker_specs in kv_cache_specs:
        for layer_name, layer_spec in worker_specs.items():
            if layer_name in merged_kv_cache_specs:
                assert merged_kv_cache_specs[layer_name] == layer_spec
            else:
                merged_kv_cache_specs[layer_name] = layer_spec

    global_groups = _get_kv_cache_groups(
        vllm_config, merged_kv_cache_specs)
    projected_groups_per_worker = [
        kv_utils._project_kv_cache_groups_to_worker(
            global_groups, worker_specs)
        for worker_specs in kv_cache_specs
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        kv_utils._auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory)

    for groups, worker_memory in zip(
            projected_groups_per_worker, available_memory):
        if not groups:
            continue
        kv_utils._check_enough_kv_cache_memory(
            worker_memory,
            partial(
                _max_memory_usage_bytes_from_groups,
                vllm_config,
                groups,
            ),
            vllm_config.model_config.max_model_len,
            partial(
                kv_utils._estimate_max_model_len_from_groups,
                vllm_config,
                groups,
            ),
        )

    configs = []
    for groups, worker_specs, worker_memory in zip(
            projected_groups_per_worker, kv_cache_specs, available_memory):
        assert sum(len(group.layer_names) for group in groups) == len(
            worker_specs), "Some layers are not assigned to any group."
        configs.append(_get_kv_cache_config_from_groups(
            vllm_config, groups, worker_memory))

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
