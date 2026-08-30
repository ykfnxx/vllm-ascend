# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Mapping
from dataclasses import replace

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)


def add_decode_external_main_cache(
    kv_cache_config: KVCacheConfig,
    main_specs: Mapping[str, AscendMLAAttentionSpec],
    num_hot_blocks: int,
) -> int | None:
    """Restore worker-only Main metadata and fixed Hot Cache tensors.

    A DSA Offload Decode scheduler must budget only the cache that grows with
    sequence length (the Indexer cache). Main MLA history lives in external
    storage, while the worker owns a fixed Hot Cache row per request. The Main
    specs are therefore omitted from the scheduler KV-cache view and restored
    only on the worker-owned copy of ``kv_cache_config``.

    The Main layers are folded into the existing Indexer group so scheduler
    block-table group IDs remain unchanged. Their fixed-size tensors are added
    separately and are not visible to ``KVCacheManager``.
    """
    if not main_specs:
        return None
    if num_hot_blocks <= 0:
        raise ValueError(
            f"DSA Offload Decode Hot Cache blocks must be positive, got {num_hot_blocks}."
        )

    indexer_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if _contains_indexer_spec(group)
    ]
    if len(indexer_group_ids) != 1:
        raise ValueError(
            "DSA Offload Decode requires exactly one scheduler-managed "
            f"Indexer KV-cache group, got {indexer_group_ids}."
        )

    group_id = indexer_group_ids[0]
    indexer_group = kv_cache_config.kv_cache_groups[group_id]
    indexer_specs = _expand_group_specs(indexer_group)
    if not all(
        isinstance(spec, AscendSFAIndexerCacheSpec)
        for spec in indexer_specs.values()
    ):
        raise ValueError(
            "The DSA Offload Decode Indexer group must contain only Indexer specs."
        )

    main_layer_names = set(main_specs)
    if set(indexer_specs) & main_layer_names:
        raise ValueError(
            "DSA Offload Main layers must be absent from the scheduler KV-cache group."
        )
    if any(
        set(cache_tensor.shared_by) & main_layer_names
        for cache_tensor in kv_cache_config.kv_cache_tensors
    ):
        raise ValueError(
            "DSA Offload Main layers must not own scheduler-managed cache tensors."
        )

    block_size = indexer_group.kv_cache_spec.block_size
    if any(spec.block_size != block_size for spec in main_specs.values()):
        raise ValueError(
            "DSA Offload Main and Indexer caches must use the same block size."
        )

    combined_specs: dict[str, KVCacheSpec] = dict(indexer_specs)
    combined_specs.update(main_specs)
    kv_cache_config.kv_cache_groups[group_id] = replace(
        indexer_group,
        layer_names=[*indexer_group.layer_names, *main_specs],
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=block_size,
            kv_cache_specs=combined_specs,
        ),
    )
    kv_cache_config.kv_cache_tensors.extend(
        KVCacheTensor(
            size=num_hot_blocks * spec.page_size_bytes,
            shared_by=[layer_name],
        )
        for layer_name, spec in main_specs.items()
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
        if set(group.layer_names) != set(group_spec.kv_cache_specs):
            raise ValueError(
                "Uniform KV-cache group layer names and specs must match."
            )
        return dict(group_spec.kv_cache_specs)
    return dict.fromkeys(group.layer_names, group_spec)
