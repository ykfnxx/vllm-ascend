#!/usr/bin/env python3

"""Validate or repair the installed A5 custom-op kernel indexes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SOC = "ascend950"
OP_TYPE = "AsuKvGather"
OP_DIR = "asu_kv_gather"
NORMAL_INDEX = "binary_info_config.json"
RELOCATABLE_INDEX = "relocatable_kernel_info_config.json"


class IndexValidationError(RuntimeError):
    """Raised when an installed kernel index is stale or incomplete."""


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexValidationError(f"cannot load {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise IndexValidationError(f"{path} does not contain a JSON object")
    return data


def _expected_json_paths(kernel_root: Path, *, relocatable: bool) -> set[Path]:
    op_root = kernel_root / SOC / OP_DIR
    candidates = set()
    for path in op_root.glob(f"{OP_TYPE}_*.json"):
        is_relocatable = path.name.endswith("_relocatable.json")
        if is_relocatable == relocatable:
            candidates.add(path.relative_to(kernel_root))
    if not candidates:
        kind = "relocatable" if relocatable else "normal"
        raise IndexValidationError(f"no {kind} {OP_TYPE} JSON files under {op_root}")
    return candidates


def _resolve_index_reference(kernel_root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise IndexValidationError(f"invalid {field} reference: {value!r}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise IndexValidationError(f"unsafe {field} reference: {value}")
    resolved_root = kernel_root.resolve()
    resolved = (kernel_root / relative).resolve()
    if resolved_root not in resolved.parents:
        raise IndexValidationError(f"{field} escapes kernel root: {value}")
    if not resolved.is_file():
        raise IndexValidationError(f"referenced {field} is missing: {resolved}")
    return relative


def _validate_one_index(
    kernel_root: Path,
    index_path: Path,
    *,
    relocatable: bool,
) -> int:
    expected_jsons = _expected_json_paths(
        kernel_root,
        relocatable=relocatable,
    )
    data = _load_json(index_path)
    op_config = data.get(OP_TYPE)
    if not isinstance(op_config, dict):
        raise IndexValidationError(f"{index_path} does not contain {OP_TYPE}")
    binary_list = op_config.get("binaryList")
    if not isinstance(binary_list, list) or not binary_list:
        raise IndexValidationError(f"{index_path} has no {OP_TYPE} binaryList")

    referenced_jsons = set()
    for entry in binary_list:
        if not isinstance(entry, dict):
            raise IndexValidationError(f"invalid {OP_TYPE} binaryList entry in {index_path}")
        referenced_jsons.add(
            _resolve_index_reference(
                kernel_root,
                entry.get("jsonPath"),
                "jsonPath",
            )
        )
        _resolve_index_reference(
            kernel_root,
            entry.get("binPath"),
            "binPath",
        )

    if referenced_jsons != expected_jsons:
        missing = sorted(str(path) for path in expected_jsons - referenced_jsons)
        extra = sorted(str(path) for path in referenced_jsons - expected_jsons)
        raise IndexValidationError(f"{index_path} {OP_TYPE} references mismatch; missing={missing} extra={extra}")
    return len(binary_list)


def validate(kernel_root: Path) -> tuple[int, int]:
    kernel_root = kernel_root.resolve()
    config_root = kernel_root / "config" / SOC
    normal_count = _validate_one_index(
        kernel_root,
        config_root / NORMAL_INDEX,
        relocatable=False,
    )
    relocatable_count = _validate_one_index(
        kernel_root,
        config_root / RELOCATABLE_INDEX,
        relocatable=True,
    )
    return normal_count, relocatable_count


def _load_index_generator(repo_root: Path):
    generator_dir = repo_root / "csrc/cmake/scripts/util"
    generator_path = generator_dir / "ascendc_ops_config.py"
    if not generator_path.is_file():
        raise IndexValidationError(f"kernel index generator is missing: {generator_path}")
    module_name = "a5_kvgather_ascendc_ops_config"
    spec = importlib.util.spec_from_file_location(module_name, generator_path)
    if spec is None or spec.loader is None:
        raise IndexValidationError(f"cannot load kernel index generator: {generator_path}")
    sys.path.insert(0, str(generator_dir))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _generate_configs(repo_root: Path, kernel_root: Path, output_dir: Path) -> None:
    generator = _load_index_generator(repo_root)
    soc_root = kernel_root / SOC
    all_jsons = generator.get_specified_suffix_file(str(soc_root), ".json")
    relocatable_jsons = {path for path in all_jsons if path.endswith("_relocatable.json")}
    normal_jsons = sorted(set(all_jsons) - relocatable_jsons)
    relocatable_jsons = sorted(relocatable_jsons)
    if not normal_jsons or not relocatable_jsons:
        raise IndexValidationError("installed custom-op package has no normal or relocatable kernel JSON files")

    normal_config = {NORMAL_INDEX: {}}
    for path in normal_jsons:
        generator.gen_ops_config(path, SOC, NORMAL_INDEX, normal_config)
    generator.write_jsons(output_dir, normal_config.keys(), normal_config)

    relocatable_config = {RELOCATABLE_INDEX: {}}
    for path in relocatable_jsons:
        generator.gen_ops_config(
            path,
            SOC,
            RELOCATABLE_INDEX,
            relocatable_config,
        )
    generator.write_jsons(
        output_dir,
        [RELOCATABLE_INDEX],
        relocatable_config,
    )


def repair(
    repo_root: Path,
    kernel_root: Path,
    backup_dir: Path,
) -> tuple[int, int]:
    repo_root = repo_root.resolve()
    kernel_root = kernel_root.resolve()
    config_root = kernel_root / "config" / SOC
    if not config_root.is_dir():
        raise IndexValidationError(f"kernel config directory is missing: {config_root}")
    if backup_dir.exists():
        raise IndexValidationError(f"backup path already exists: {backup_dir}")
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config_root, backup_dir)

    with tempfile.TemporaryDirectory(
        prefix=".a5-kvgather-index.",
        dir=config_root.parent,
    ) as stage_text:
        stage = Path(stage_text)
        _generate_configs(repo_root, kernel_root, stage)
        _validate_one_index(
            kernel_root,
            stage / NORMAL_INDEX,
            relocatable=False,
        )
        _validate_one_index(
            kernel_root,
            stage / RELOCATABLE_INDEX,
            relocatable=True,
        )

        for source in sorted(stage.iterdir()):
            if not source.is_file() or source.suffix != ".json":
                continue
            destination = config_root / source.name
            temporary = config_root / f".{source.name}.a5-kvgather-{os.getpid()}.tmp"
            shutil.copy2(source, temporary)
            temporary.chmod(0o640)
            os.replace(temporary, destination)

    return validate(kernel_root)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-root", type=Path, required=True)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        normal_count, relocatable_count = validate(args.kernel_root)
        print(f"A5_CUSTOM_OP_KERNEL_INDEX_READY: op={OP_TYPE} normal={normal_count} relocatable={relocatable_count}")
        return 0
    except IndexValidationError as exc:
        if not args.repair:
            print(f"A5_CUSTOM_OP_KERNEL_INDEX_INVALID: {exc}", file=sys.stderr)
            return 1
        if args.repo_root is None or args.backup_dir is None:
            print(
                "A5_CUSTOM_OP_KERNEL_INDEX_FAILED: --repair requires --repo-root and --backup-dir",
                file=sys.stderr,
            )
            return 2
        print(f"A5_CUSTOM_OP_KERNEL_INDEX_STALE: {exc}")

    try:
        normal_count, relocatable_count = repair(
            args.repo_root,
            args.kernel_root,
            args.backup_dir,
        )
    except (IndexValidationError, OSError, ValueError) as exc:
        print(f"A5_CUSTOM_OP_KERNEL_INDEX_FAILED: {exc}", file=sys.stderr)
        return 2

    print(
        "A5_CUSTOM_OP_KERNEL_INDEX_REPAIRED: "
        f"op={OP_TYPE} normal={normal_count} relocatable={relocatable_count} "
        f"backup={args.backup_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
