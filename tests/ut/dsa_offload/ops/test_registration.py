# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OPERATOR_NAMES = (
    "dsa_offload_lookup_update_batch",
    "asu_kv_gather",
)


def test_native_sources_use_only_dsa_offload_names() -> None:
    operator_roots = [ROOT / "csrc" / "attention" / operator_name for operator_name in OPERATOR_NAMES]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for operator_root in operator_roots
        for path in operator_root.rglob("*")
        if path.is_file()
    )

    assert "dsa_sparse" not in source
    assert "DsaSparse" not in source
    assert "DSA_SPARSE" not in source
    assert all(operator_root.is_dir() for operator_root in operator_roots)
    assert not (ROOT / "csrc" / "attention" / "dsa_offload_lookup_update").exists()


def test_native_operator_directories_match_kernel_names() -> None:
    for operator_name in OPERATOR_NAMES:
        operator_root = ROOT / "csrc" / "attention" / operator_name
        kernel_source = operator_root / "op_kernel" / f"{operator_name}.cpp"

        assert operator_root.name == operator_name
        assert kernel_source.is_file()


def test_torch_and_meta_registrations_exist() -> None:
    binding = (ROOT / "csrc" / "torch_binding.cpp").read_text(encoding="utf-8")
    meta = (ROOT / "csrc" / "torch_binding_meta.cpp").read_text(encoding="utf-8")

    for operator_name in OPERATOR_NAMES:
        assert f'"{operator_name}(' in binding
        assert f'ops.impl(\n        "{operator_name}"' in binding
        assert f'ops.impl("{operator_name}"' in meta


def test_only_a5_build_selects_dsa_offload() -> None:
    build_script = (ROOT / "csrc" / "build_aclnn.sh").read_text(encoding="utf-8")
    a5_branch = build_script.split('elif [[ "$SOC_VERSION" =~ ^ascend950 ]]', 1)[1].split("else", 1)[0]
    earlier_branches = build_script.split('elif [[ "$SOC_VERSION" =~ ^ascend950 ]]', 1)[0]

    for operator_name in OPERATOR_NAMES:
        assert f'"{operator_name}"' in a5_branch
        assert f'"{operator_name}"' not in earlier_branches
