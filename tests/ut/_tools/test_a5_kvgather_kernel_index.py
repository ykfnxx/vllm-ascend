# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts/a5_kvgather_kernel_index.py"
SPEC = importlib.util.spec_from_file_location(
    "a5_kvgather_kernel_index",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
INDEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INDEX)


def _write_kernel(root: Path, name: str, *, relocatable: bool) -> None:
    suffix = "_relocatable" if relocatable else ""
    kernel_dir = root / "ascend950/asu_kv_gather"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    base = f"AsuKvGather_{name}{suffix}"
    metadata = {
        "binFileName": base,
        "binFileSuffix": ".o",
        "coreType": "MIX",
        "taskRation": "tilingKey",
        "supportInfo": {
            "simplifiedKey": f"key-{name}",
            "staticKey": f"static-{name}",
            "inputs": [],
            "outputs": [],
        },
    }
    (kernel_dir / f"{base}.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (kernel_dir / f"{base}.o").write_bytes(b"kernel")


def test_repair_rebuilds_stale_aggregate_indexes(tmp_path: Path) -> None:
    kernel_root = tmp_path / "kernel"
    config_root = kernel_root / "config/ascend950"
    config_root.mkdir(parents=True)
    (config_root / "binary_info_config.json").write_text("{}", encoding="utf-8")
    (config_root / "relocatable_kernel_info_config.json").write_text(
        "{}",
        encoding="utf-8",
    )
    _write_kernel(kernel_root, "normal", relocatable=False)
    _write_kernel(kernel_root, "reloc", relocatable=True)

    with pytest.raises(INDEX.IndexValidationError):
        INDEX.validate(kernel_root)

    backup = tmp_path / "index-backup"
    assert INDEX.repair(REPO_ROOT, kernel_root, backup) == (1, 1)
    assert json.loads((backup / "binary_info_config.json").read_text()) == {}
    assert INDEX.validate(kernel_root) == (1, 1)


def test_validation_rejects_missing_referenced_binary(tmp_path: Path) -> None:
    kernel_root = tmp_path / "kernel"
    config_root = kernel_root / "config/ascend950"
    config_root.mkdir(parents=True)
    _write_kernel(kernel_root, "normal", relocatable=False)
    _write_kernel(kernel_root, "reloc", relocatable=True)
    INDEX.repair(REPO_ROOT, kernel_root, tmp_path / "index-backup")

    (kernel_root / "ascend950/asu_kv_gather/AsuKvGather_normal.o").unlink()
    with pytest.raises(INDEX.IndexValidationError, match="binPath is missing"):
        INDEX.validate(kernel_root)
