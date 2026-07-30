# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LOOKUP_ROOT = (
    ROOT / "csrc" / "attention" / "asu_hbm_index_lookup"
)
MAINTAIN_ROOT = (
    ROOT
    / "csrc"
    / "attention"
    / "asu_hbm_index_maintain_aicpu"
)
BUILD_SCRIPT = ROOT / "csrc" / "build_aclnn.sh"
BINDING_SOURCE = ROOT / "csrc" / "torch_binding.cpp"
META_SOURCE = ROOT / "csrc" / "torch_binding_meta.cpp"


def _soc_branch(source: str, name: str, next_name: str) -> str:
    start = source.index(
        f'elif [[ "$SOC_VERSION" =~ ^{name}'
    )
    end = source.index(
        f'elif [[ "$SOC_VERSION" =~ ^{next_name}',
        start,
    )
    return source[start:end]


def test_a3_package_selects_both_legacy_index_operators() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    a2_branch = _soc_branch(source, "ascend910b", "ascend910_93")
    a3_branch = _soc_branch(source, "ascend910_93", "ascend950")
    a5_branch = source[source.index(
        'elif [[ "$SOC_VERSION" =~ ^ascend950'
    ):]

    for op_name in (
        "asu_hbm_index_lookup",
        "asu_hbm_index_maintain_aicpu",
    ):
        assert f'"{op_name}"' in a3_branch
        assert f'"{op_name}"' not in a2_branch
        assert f'"{op_name}"' not in a5_branch

    assert '"dsa_sparse_lookup_update"' in a5_branch
    assert '"dsa_sparse_lookup_update"' not in a3_branch


def test_lookup_is_registered_only_for_ascend910_93() -> None:
    source = (
        LOOKUP_ROOT
        / "op_host"
        / "asu_hbm_index_lookup_def.cpp"
    ).read_text(encoding="utf-8")

    assert 'AddConfig("ascend910_93"' in source
    assert 'AddConfig("ascend910b"' not in source
    assert 'AddConfig("ascend950"' not in source


def test_lookup_exposes_v023_public_aclnn_wrapper() -> None:
    cmake_source = (
        LOOKUP_ROOT / "op_host" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")
    source = (
        LOOKUP_ROOT
        / "op_host"
        / "op_api"
        / "aclnn_asu_hbm_index_lookup.cpp"
    ).read_text(encoding="utf-8")

    assert (
        "aclnnAsuHbmIndexLookupGetWorkspaceSize"
        in source
    )
    assert (
        "aclnnInnerAsuHbmIndexLookupGetWorkspaceSize"
        in source
    )
    assert "aclnnAsuHbmIndexLookup(" in source
    assert "aclnnInnerAsuHbmIndexLookup(" in source
    assert "target_sources(op_host_aclnnInner" in cmake_source
    assert "ACLNNTYPE aclnn_inner" in cmake_source


def test_lookup_preserves_a3_kernel_compile_mode() -> None:
    source = (
        LOOKUP_ROOT / "op_host" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "--cce-auto-sync=off" in source
    assert "--op_relocatable_kernel_binary=true" in source


def test_maintain_uses_v023_unified_aicpu_package_layout() -> None:
    cmake_source = (
        MAINTAIN_ROOT / "CMakeLists.txt"
    ).read_text(encoding="utf-8")
    l0_source = (
        MAINTAIN_ROOT
        / "op_api"
        / "l0_asu_hbm_index_maintain_aicpu.cpp"
    ).read_text(encoding="utf-8")
    kernel_source = (
        MAINTAIN_ROOT
        / "op_kernel_aicpu"
        / "asu_hbm_index_maintain_aicpu_aicpu.cpp"
    ).read_text(encoding="utf-8")
    json_source = (
        MAINTAIN_ROOT
        / "op_kernel_aicpu"
        / "asu_hbm_index_maintain_aicpu_aicpu.json"
    ).read_text(encoding="utf-8")

    assert "add_modules_sources_aicpu(" in cmake_source
    assert "AicpuTaskSpace" in l0_source
    for output_index in range(4):
        assert f"space.SetRef({output_index})" in l0_source
    assert "REGISTER_CPU_KERNEL(" in kernel_source
    assert '"kernelSo": "libtransformer_aicpu_kernels.so"' in json_source
    assert not (MAINTAIN_ROOT / "cpukernel").exists()


def test_maintain_preserves_lookup_results_during_eviction() -> None:
    source = (
        MAINTAIN_ROOT
        / "op_kernel_aicpu"
        / "asu_hbm_index_maintain_aicpu_aicpu.cpp"
    ).read_text(encoding="utf-8")

    assert "last_query_slots + req_id * QUERY_COUNT" in source
    assert "MarkProtectedSlot(" in source
    assert "!IsProtectedSlot(protected_slots, slot)" in source
    assert "req_slot_to_index[slot] = NOT_FOUND" in source
    assert "req_free_slots[static_cast<uint32_t>(head)]" in source
    assert "free_head[pool_entry * FREE_HEAD_STRIDE] = head" in source


def test_torch_and_meta_register_both_a3_operators() -> None:
    binding_source = BINDING_SOURCE.read_text(encoding="utf-8")
    meta_source = META_SOURCE.read_text(encoding="utf-8")

    lookup_schema_start = binding_source.index(
        '"asu_hbm_index_lookup("'
    )
    lookup_schema_end = binding_source.index(
        'ops.impl(\n        "asu_hbm_index_lookup"',
        lookup_schema_start,
    )
    lookup_schema = binding_source[
        lookup_schema_start:lookup_schema_end
    ]
    for name in (
        "index",
        "slot_to_index",
        "free_slots",
        "free_head",
        "req_pool_entries",
        "query_index",
        "lookup_mask",
        "req_num",
    ):
        assert name in lookup_schema
    assert "-> (Tensor, Tensor)" in lookup_schema

    maintain_schema_start = binding_source.index(
        '"asu_hbm_index_maintain_aicpu("'
    )
    maintain_schema_end = binding_source.index(
        'ops.impl(\n        "asu_hbm_index_maintain_aicpu"',
        maintain_schema_start,
    )
    maintain_schema = binding_source[
        maintain_schema_start:maintain_schema_end
    ]
    assert "last_query_slots" in maintain_schema
    assert "int seed" in maintain_schema
    assert "-> ()" in maintain_schema

    assert (
        'ops.impl("asu_hbm_index_lookup",'
        in meta_source
    )
    assert (
        'ops.impl("asu_hbm_index_maintain_aicpu",'
        in meta_source
    )
