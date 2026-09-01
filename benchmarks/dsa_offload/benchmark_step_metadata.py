# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import argparse
import hashlib
import importlib.util
import json
import pickle
import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT / "vllm_ascend" / "dsa_offload" / "metadata.py"


def load_metadata_module():
    for package_name in ("vllm_ascend", "vllm_ascend.dsa_offload"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
    name = "vllm_ascend.dsa_offload.metadata"
    spec = importlib.util.spec_from_file_location(name, METADATA_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(function, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        begin = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - begin)
    return statistics.median(samples) / 1_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--history-blocks", type=int, default=1024)
    parser.add_argument("--delta-blocks", type=int, default=1)
    parser.add_argument("--mtp-candidates", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    metadata = load_metadata_module()
    canonical = [
        hashlib.sha256(index.to_bytes(8, "big")).digest()
        for index in range(args.history_blocks + args.delta_blocks)
    ]
    history = canonical[: args.history_blocks]
    appended = canonical[
        args.history_blocks : args.history_blocks + args.delta_blocks
    ]
    candidate = canonical[: args.mtp_candidates]
    legacy_payload = {
        "committed": {
            f"request-{request}": list(history + appended)
            for request in range(args.requests)
        },
        "candidate": {
            f"request-{request}": list(candidate)
            for request in range(args.requests)
        },
    }
    appended_keys = tuple(metadata.make_block_key(value) for value in appended)
    candidate_keys = tuple(metadata.make_block_key(value) for value in candidate)
    delta_payload = metadata.DSAOffloadStepMetadata(
        committed_updates={
            f"request-{request}": (
                args.history_blocks,
                appended_keys,
            )
            for request in range(args.requests)
        },
        decode_contexts={},
        candidate_keys={
            f"request-{request}": candidate_keys
            for request in range(args.requests)
        },
    )
    legacy_bytes = pickle.dumps(legacy_payload, protocol=pickle.HIGHEST_PROTOCOL)
    delta_bytes = pickle.dumps(delta_payload, protocol=pickle.HIGHEST_PROTOCOL)

    committed_seed = [
        metadata.make_block_key(value) for value in history
    ]

    committed = {
        request_id: committed_seed.copy()
        for request_id in delta_payload.committed_updates
    }

    def apply_delta() -> None:
        for request_id, update in delta_payload.committed_updates.items():
            metadata.apply_committed_update(
                request_id,
                committed[request_id],
                update,
            )
            del committed[request_id][args.history_blocks :]

    result = {
        "requests": args.requests,
        "history_blocks": args.history_blocks,
        "delta_blocks": args.delta_blocks,
        "mtp_candidates": args.mtp_candidates,
        "legacy_pickle_bytes": len(legacy_bytes),
        "delta_pickle_bytes": len(delta_bytes),
        "legacy_encode_us_p50": measure(
            lambda: pickle.dumps(
                legacy_payload,
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
            args.iterations,
        ),
        "delta_encode_us_p50": measure(
            lambda: pickle.dumps(
                delta_payload,
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
            args.iterations,
        ),
        "legacy_decode_us_p50": measure(
            lambda: pickle.loads(legacy_bytes),
            args.iterations,
        ),
        "delta_decode_us_p50": measure(
            lambda: pickle.loads(delta_bytes),
            args.iterations,
        ),
        "worker_apply_us_p50": measure(
            apply_delta,
            args.iterations,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
