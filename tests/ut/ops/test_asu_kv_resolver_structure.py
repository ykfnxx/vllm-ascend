from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_asu_kv_resolver_uses_ascend_op_directory_structure():
    repo_root = _repo_root()
    op_root = repo_root / "csrc" / "asu_kv_resolver"

    expected_files = [
        op_root / "asu_kv_resolver_torch_adpt.h",
        op_root / "op_host" / "CMakeLists.txt",
        op_root / "op_host" / "asu_kv_resolver_def.cpp",
        op_root / "op_host" / "asu_kv_resolver_infershape.cpp",
        op_root / "op_host" / "asu_kv_resolver_tiling.cpp",
        op_root / "op_host" / "asu_kv_resolver_tiling.h",
        op_root / "op_kernel" / "asu_kv_resolver.cpp",
    ]
    for path in expected_files:
        assert path.exists(), f"{path.relative_to(repo_root)} is missing"

    assert not (repo_root / "csrc" / "kernels" /
                "asu_kv_resolver.cpp").exists()


def test_asu_kv_resolver_torch_adapter_calls_aclnn_op():
    repo_root = _repo_root()
    adapter = (
        repo_root / "csrc" / "asu_kv_resolver" /
        "asu_kv_resolver_torch_adpt.h"
    ).read_text()

    assert "EXEC_NPU_CMD(" in adapter
    assert "aclnnAsuKvResolver" in adapter
    assert "SetCustomHandler" not in adapter
