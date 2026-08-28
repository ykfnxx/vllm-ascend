# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

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


@dataclass(frozen=True)
class DSASparseExternalMainSpecs:
    """Worker-local Main specs omitted from the Decode scheduler view."""

    by_layer: Mapping[str, AscendMLAAttentionSpec]

    @classmethod
    def empty(cls) -> "DSASparseExternalMainSpecs":
        return cls(MappingProxyType({}))

    @classmethod
    def from_mapping(
        cls,
        specs: Mapping[str, AscendMLAAttentionSpec],
    ) -> "DSASparseExternalMainSpecs":
        return cls(MappingProxyType(dict(specs)))

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(self.by_layer)

    def __bool__(self) -> bool:
        return bool(self.by_layer)


def add_external_main_metadata(
    kv_cache_config: KVCacheConfig,
    external_main_specs: DSASparseExternalMainSpecs,
) -> int | None:
    """Add runner-only Main metadata to the sole Indexer cache group.

    ``kv_cache_config`` must already be a worker-owned copy. Its cache tensors
    remain untouched, so only the Indexer payload is allocated and registered.
    The returned group id is the existing Indexer group position.
    """

    if not external_main_specs:
        return None

    indexer_group_ids = [
        group_id for group_id, group in enumerate(kv_cache_config.kv_cache_groups) if _contains_indexer_spec(group)
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
    assert not (set(indexer_specs) & set(external_main_specs.by_layer)), (
        "External Main layers must be absent from the scheduler KV cache group."
    )
    assert not any(
        set(cache_tensor.shared_by) & set(external_main_specs.by_layer)
        for cache_tensor in kv_cache_config.kv_cache_tensors
    ), "External Main layers must not own scheduler cache tensors."

    block_size = indexer_group.kv_cache_spec.block_size
    assert all(spec.block_size == block_size for spec in external_main_specs.by_layer.values()), (
        "Main and Indexer specs in a DSA Sparse cohort must use one block size."
    )

    combined_specs: dict[str, KVCacheSpec] = dict(indexer_specs)
    combined_specs.update(external_main_specs.by_layer)
    kv_cache_config.kv_cache_groups[group_id] = replace(
        indexer_group,
        layer_names=[
            *indexer_group.layer_names,
            *external_main_specs.layer_names,
        ],
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=block_size,
            kv_cache_specs=combined_specs,
        ),
    )
    return group_id


def _contains_indexer_spec(group: KVCacheGroupSpec) -> bool:
    return any(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in _expand_group_specs(group).values())


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
