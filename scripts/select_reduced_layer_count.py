#!/usr/bin/env python3
"""Select the smallest GLM prefix that still exercises the MoE path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--minimum-moe-layers", type=int, default=1)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    total_layers = int(config.get("num_hidden_layers", 0))
    first_dense = int(config.get("first_k_dense_replace", 0))
    moe_frequency = int(config.get("moe_layer_freq", 1))
    routed_experts = config.get("n_routed_experts")
    if total_layers <= 0:
        raise ValueError("config.json has no valid num_hidden_layers")
    if routed_experts is None or int(routed_experts) <= 0:
        raise ValueError("config.json does not describe a routed-expert model")
    if moe_frequency <= 0:
        raise ValueError("moe_layer_freq must be positive")
    if args.minimum_moe_layers <= 0:
        raise ValueError("minimum-moe-layers must be positive")

    moe_layers = [
        layer
        for layer in range(total_layers)
        if layer >= first_dense and layer % moe_frequency == 0
    ]
    if len(moe_layers) < args.minimum_moe_layers:
        raise ValueError(
            f"model has only {len(moe_layers)} MoE layers; "
            f"requested {args.minimum_moe_layers}"
        )

    selected_layers = (
        args.layers
        if args.layers is not None
        else moe_layers[args.minimum_moe_layers - 1] + 1
    )
    if not 1 <= selected_layers <= total_layers:
        raise ValueError(
            f"layers must be between 1 and {total_layers}, got {selected_layers}"
        )
    selected_moe_layers = [layer for layer in moe_layers if layer < selected_layers]
    if len(selected_moe_layers) < args.minimum_moe_layers:
        raise ValueError(
            f"{selected_layers} layers contain only {len(selected_moe_layers)} MoE "
            f"layers; at least {args.minimum_moe_layers} required. "
            f"The first MoE layer is {moe_layers[0]}."
        )
    print(selected_layers)


if __name__ == "__main__":
    main()
