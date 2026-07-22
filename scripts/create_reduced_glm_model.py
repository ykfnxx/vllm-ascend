#!/usr/bin/env python3
"""Create a small GLM checkpoint for single-card DMP validation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch  # noqa: F401 - safetensors uses the torch backend
from safetensors import safe_open
from safetensors.torch import save_file

LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth")
REDUCED_MODEL_FORMAT_VERSION = 2
EXCLUDED_GLOBAL_WEIGHTS = {"rot.weight"}


def keep_weight(name: str, num_layers: int) -> bool:
    if name in EXCLUDED_GLOBAL_WEIGHTS:
        return False
    match = LAYER_PATTERN.search(name)
    return match is None or int(match.group(1)) < num_layers


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def copy_metadata(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith(".safetensors.index.json"):
            continue
        if any(path.match(pattern) for pattern in WEIGHT_PATTERNS):
            continue
        # Do not accidentally copy backup/checkpoint variants of large weights.
        if ".safetensors." in path.name or ".bin." in path.name:
            continue
        shutil.copy2(path, destination / path.name)


def write_reduced_config(source: Path, destination: Path, num_layers: int) -> int:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    original_layers = int(config.get("num_hidden_layers", 0))
    if original_layers <= 0:
        raise ValueError("config.json has no valid num_hidden_layers")
    if not 1 <= num_layers <= original_layers:
        raise ValueError(
            f"layers must be between 1 and {original_layers}, got {num_layers}"
        )
    config["num_hidden_layers"] = num_layers
    if "num_nextn_predict_layers" in config:
        config["num_nextn_predict_layers"] = 0
    (destination / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return original_layers


def reduce_safetensors(
    source: Path, destination: Path, num_layers: int
) -> tuple[dict[str, str], int, int]:
    source_files = sorted(source.glob("*.safetensors"))
    if not source_files:
        raise FileNotFoundError(f"No safetensors weights found in {source}")

    weight_map: dict[str, str] = {}
    total_size = 0
    tensor_count = 0
    for shard_index, source_file in enumerate(source_files, start=1):
        with safe_open(source_file, framework="pt", device="cpu") as handle:
            available_keys = handle.keys()
            selected_keys = [
                key for key in available_keys if keep_weight(key, num_layers)
            ]
            if not selected_keys:
                continue
            print(
                f"[{shard_index}/{len(source_files)}] "
                f"{source_file.name}: keeping {len(selected_keys)} tensors",
                flush=True,
            )
            tensors = {key: handle.get_tensor(key) for key in selected_keys}
            metadata = handle.metadata()
            output_file = destination / source_file.name
            save_file(tensors, output_file, metadata=metadata)
            for key, tensor in tensors.items():
                weight_map[key] = output_file.name
                total_size += tensor_bytes(tensor)
                tensor_count += 1
            del tensors

    if not weight_map:
        raise RuntimeError("No weights matched the reduced-model filter")
    return weight_map, total_size, tensor_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    marker = output / "DMP_REDUCED_MODEL.json"
    if marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if (
            int(existing.get("format_version", -1))
            != REDUCED_MODEL_FORMAT_VERSION
            or int(existing.get("num_hidden_layers", -1)) != args.layers
        ):
            raise RuntimeError(
                f"Existing model at {output} is stale or has a different layer count"
            )
        print(f"Reduced model is already ready: {output}")
        return
    if output.exists():
        raise RuntimeError(
            f"Output already exists but is incomplete: {output}. "
            "Remove or rename it, then retry."
        )

    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        copy_metadata(source, temporary)
        original_layers = write_reduced_config(source, temporary, args.layers)
        weight_map, total_size, tensor_count = reduce_safetensors(
            source, temporary, args.layers
        )
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        marker_data = {
            "format_version": REDUCED_MODEL_FORMAT_VERSION,
            "source": str(source),
            "original_num_hidden_layers": original_layers,
            "num_hidden_layers": args.layers,
            "tensor_count": tensor_count,
            "weight_bytes": total_size,
            "excluded_global_weights": sorted(EXCLUDED_GLOBAL_WEIGHTS),
        }
        (temporary / marker.name).write_text(
            json.dumps(marker_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        print(f"Incomplete temporary directory retained for diagnosis: {temporary}")
        raise

    print(f"Reduced model ready: {output}")
    print(f"Layers: {args.layers}/{original_layers}")
    print(f"Selected tensors: {tensor_count}")
    print(f"Selected weight bytes: {total_size}")


if __name__ == "__main__":
    main()
