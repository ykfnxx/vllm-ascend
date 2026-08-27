# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def test_native_sources_use_only_dsa_offload_names() -> None:
    operator_root = ROOT / "csrc" / "attention" / "dsa_offload"
    source = "\n".join(path.read_text(encoding="utf-8") for path in operator_root.rglob("*") if path.is_file())

    assert "dsa_sparse" not in source
    assert "DsaSparse" not in source
    assert "DSA_SPARSE" not in source
    assert (operator_root / "lookup_update").is_dir()
    assert (operator_root / "lookup_update_batch").is_dir()


def test_torch_and_meta_registrations_exist() -> None:
    binding = (ROOT / "csrc" / "torch_binding.cpp").read_text(encoding="utf-8")
    meta = (ROOT / "csrc" / "torch_binding_meta.cpp").read_text(encoding="utf-8")

    for operator_name in ("dsa_offload_lookup_update", "dsa_offload_lookup_update_batch"):
        assert f'"{operator_name}(' in binding
        assert f'ops.impl(\n        "{operator_name}"' in binding
        assert f'ops.impl("{operator_name}"' in meta


def test_only_a5_build_selects_dsa_offload() -> None:
    build_script = (ROOT / "csrc" / "build_aclnn.sh").read_text(encoding="utf-8")
    a5_branch = build_script.split('elif [[ "$SOC_VERSION" =~ ^ascend950 ]]', 1)[1].split("else", 1)[0]
    earlier_branches = build_script.split('elif [[ "$SOC_VERSION" =~ ^ascend950 ]]', 1)[0]

    assert '"dsa_offload"' in a5_branch
    assert '"dsa_offload"' not in earlier_branches
