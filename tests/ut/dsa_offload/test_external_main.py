# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_external_main_module(monkeypatch):
    @dataclass(frozen=True)
    class KVCacheSpec:
        block_size: int
        page_size_bytes: int

    @dataclass(frozen=True)
    class AscendMLAAttentionSpec(KVCacheSpec):
        pass

    @dataclass(frozen=True)
    class AscendSFAIndexerCacheSpec(KVCacheSpec):
        pass

    @dataclass(frozen=True)
    class HiddenStateCacheSpec(KVCacheSpec):
        pass

    @dataclass(frozen=True)
    class UniformTypeKVCacheSpecs:
        block_size: int
        kv_cache_specs: dict[str, KVCacheSpec]

    @dataclass(frozen=True)
    class KVCacheGroupSpec:
        layer_names: list[str]
        kv_cache_spec: KVCacheSpec | UniformTypeKVCacheSpecs

    @dataclass
    class KVCacheTensor:
        size: int
        shared_by: list[str]

    @dataclass
    class KVCacheConfig:
        num_blocks: int
        kv_cache_tensors: list[KVCacheTensor] = field(default_factory=list)
        kv_cache_groups: list[KVCacheGroupSpec] = field(default_factory=list)

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.kv_cache_interface": types.ModuleType(
            "vllm.v1.kv_cache_interface"
        ),
        "vllm_ascend": types.ModuleType("vllm_ascend"),
        "vllm_ascend.core": types.ModuleType("vllm_ascend.core"),
        "vllm_ascend.core.kv_cache_interface": types.ModuleType(
            "vllm_ascend.core.kv_cache_interface"
        ),
        "vllm_ascend.dsa_offload": types.ModuleType(
            "vllm_ascend.dsa_offload"
        ),
    }
    interface = modules["vllm.v1.kv_cache_interface"]
    interface.KVCacheConfig = KVCacheConfig
    interface.KVCacheGroupSpec = KVCacheGroupSpec
    interface.KVCacheSpec = KVCacheSpec
    interface.KVCacheTensor = KVCacheTensor
    interface.HiddenStateCacheSpec = HiddenStateCacheSpec
    interface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
    ascend_interface = modules[
        "vllm_ascend.core.kv_cache_interface"
    ]
    ascend_interface.AscendMLAAttentionSpec = AscendMLAAttentionSpec
    ascend_interface.AscendSFAIndexerCacheSpec = (
        AscendSFAIndexerCacheSpec
    )
    for module in modules.values():
        module.__path__ = []
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "vllm_ascend.dsa_offload._external_main_test",
        ROOT / "vllm_ascend" / "dsa_offload" / "external_main.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return types.SimpleNamespace(
        module=module,
        Config=KVCacheConfig,
        Group=KVCacheGroupSpec,
        Tensor=KVCacheTensor,
        MainSpec=AscendMLAAttentionSpec,
        IndexerSpec=AscendSFAIndexerCacheSpec,
        HiddenSpec=HiddenStateCacheSpec,
        UniformSpec=UniformTypeKVCacheSpecs,
    )


def test_decode_main_is_fixed_and_invisible_to_scheduler_budget(
    monkeypatch,
) -> None:
    api = load_external_main_module(monkeypatch)
    indexer = api.IndexerSpec(block_size=16, page_size_bytes=64)
    main = api.MainSpec(block_size=16, page_size_bytes=256)
    config = api.Config(
        num_blocks=100,
        kv_cache_tensors=[
            api.Tensor(size=6400, shared_by=["indexer"]),
        ],
        kv_cache_groups=[
            api.Group(layer_names=["indexer"], kv_cache_spec=indexer),
        ],
    )

    group_id = api.module.add_decode_external_main_cache(
        config,
        {"main": main},
        num_hot_blocks=10,
    )

    assert group_id == 0
    assert config.num_blocks == 100
    assert set(config.kv_cache_groups[0].layer_names) == {
        "indexer",
        "main",
    }
    assert isinstance(
        config.kv_cache_groups[0].kv_cache_spec,
        api.UniformSpec,
    )
    assert config.kv_cache_tensors[-1] == api.Tensor(
        size=2560,
        shared_by=["main"],
    )


def test_decode_main_requires_one_indexer_group(monkeypatch) -> None:
    api = load_external_main_module(monkeypatch)
    main = api.MainSpec(block_size=16, page_size_bytes=256)
    config = api.Config(num_blocks=100)

    with pytest.raises(ValueError, match="exactly one"):
        api.module.add_decode_external_main_cache(
            config,
            {"main": main},
            num_hot_blocks=10,
        )


def test_decode_main_preserves_mtp_specs_in_indexer_group(
    monkeypatch,
) -> None:
    api = load_external_main_module(monkeypatch)
    indexer = api.IndexerSpec(block_size=16, page_size_bytes=64)
    hidden = api.HiddenSpec(block_size=16, page_size_bytes=128)
    draft_main = api.MainSpec(block_size=16, page_size_bytes=256)
    main = api.MainSpec(block_size=16, page_size_bytes=256)
    config = api.Config(
        num_blocks=100,
        kv_cache_tensors=[
            api.Tensor(size=6400, shared_by=["indexer"]),
            api.Tensor(size=12800, shared_by=["mtp_hidden"]),
            api.Tensor(size=25600, shared_by=["mtp_main"]),
        ],
        kv_cache_groups=[
            api.Group(
                layer_names=["indexer", "mtp_hidden", "mtp_main"],
                kv_cache_spec=api.UniformSpec(
                    block_size=16,
                    kv_cache_specs={
                        "indexer": indexer,
                        "mtp_hidden": hidden,
                        "mtp_main": draft_main,
                    },
                ),
            ),
        ],
    )

    group_id = api.module.add_decode_external_main_cache(
        config,
        {"main": main},
        num_hot_blocks=10,
    )

    assert group_id == 0
    group = config.kv_cache_groups[0]
    assert set(group.layer_names) == {
        "indexer",
        "mtp_hidden",
        "mtp_main",
        "main",
    }
    assert set(group.kv_cache_spec.kv_cache_specs) == {
        "indexer",
        "mtp_hidden",
        "mtp_main",
        "main",
    }
    assert config.kv_cache_tensors[-1] == api.Tensor(
        size=2560,
        shared_by=["main"],
    )
