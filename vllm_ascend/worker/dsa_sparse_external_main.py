# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Mapping
from dataclasses import replace

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)


def add_dsa_sparse_main_metadata(
    kv_cache_config: KVCacheConfig,
    main_specs: Mapping[str, AscendMLAAttentionSpec],
) -> int | None:
    """Add runner-only Main metadata to the sole Indexer cache group.

    ``kv_cache_config`` must already be a worker-owned copy. Its cache tensors
    remain untouched, so only the Indexer payload is allocated and registered.
    The returned group id is the existing Indexer group position.
    """

    if not main_specs:
        return None

    indexer_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if _contains_indexer_spec(group)
    ]
    assert len(indexer_group_ids) == 1, (
        f"DSA Sparse Decode requires exactly one Indexer KV cache group, got {indexer_group_ids}."
    )
    group_id = indexer_group_ids[0]
    indexer_group = kv_cache_config.kv_cache_groups[group_id]
    indexer_specs = _expand_group_specs(indexer_group)

    assert all(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in indexer_specs.values()), (
        "The DSA Sparse Decode Indexer group must contain only Indexer specs."
    )
    assert not (set(indexer_specs) & set(main_specs)), (
        "External Main layers must be absent from the scheduler KV cache group."
    )
    assert not any(
        set(cache_tensor.shared_by) & set(main_specs)
        for cache_tensor in kv_cache_config.kv_cache_tensors
    ), "External Main layers must not own scheduler cache tensors."

    block_size = indexer_group.kv_cache_spec.block_size
    assert all(spec.block_size == block_size for spec in main_specs.values()), (
        "Main and Indexer specs in a DSA Sparse cohort must use one block size."
    )

    combined_specs: dict[str, KVCacheSpec] = dict(indexer_specs)
    combined_specs.update(main_specs)
    kv_cache_config.kv_cache_groups[group_id] = replace(
        indexer_group,
        layer_names=[
            *indexer_group.layer_names,
            *main_specs,
        ],
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=block_size,
            kv_cache_specs=combined_specs,
        ),
    )
    return group_id


def _contains_indexer_spec(group: KVCacheGroupSpec) -> bool:
    return any(
        isinstance(spec, AscendSFAIndexerCacheSpec)
        for spec in _expand_group_specs(group).values()
    )


def _expand_group_specs(
    group: KVCacheGroupSpec,
) -> dict[str, KVCacheSpec]:
    group_spec = group.kv_cache_spec
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        assert set(group.layer_names) == set(group_spec.kv_cache_specs), (
            "Uniform KV cache group layer names and specs must match."
        )
        return dict(group_spec.kv_cache_specs)
    return dict.fromkeys(group.layer_names, group_spec)
