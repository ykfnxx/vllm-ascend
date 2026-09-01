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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT / "vllm_ascend" / "dsa_offload" / "metadata.py"


def load_metadata_module():
    name = "dsa_offload_metadata_benchmark"
    spec = importlib.util.spec_from_file_location(name, METADATA_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def median_us(function, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    return statistics.median(samples) / 1_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--history-blocks", type=int, default=1024)
    parser.add_argument("--delta-blocks", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    metadata = load_metadata_module()
    canonical = [
        hashlib.sha256(index.to_bytes(8, "big")).digest()
        for index in range(args.history_blocks + args.delta_blocks)
    ]
    history = canonical[: args.history_blocks]
    appended = canonical[args.history_blocks :]
    legacy = {
        request_id: list(history + appended)
        for request_id in range(args.requests)
    }
    appended_keys = tuple(
        metadata.make_block_key(block_hash) for block_hash in appended
    )
    delta = metadata.DSAOffloadStepMetadata(
        committed_updates={
            str(request_id): (args.history_blocks, appended_keys)
            for request_id in range(args.requests)
        },
        decode_contexts={},
        candidate_keys={},
    )
    legacy_bytes = pickle.dumps(legacy, protocol=pickle.HIGHEST_PROTOCOL)
    delta_bytes = pickle.dumps(delta, protocol=pickle.HIGHEST_PROTOCOL)
    committed = {
        request_id: [
            metadata.make_block_key(block_hash) for block_hash in history
        ]
        for request_id in delta.committed_updates
    }

    def apply_delta() -> None:
        for request_id, update in delta.committed_updates.items():
            metadata.apply_committed_update(
                request_id,
                committed[request_id],
                update,
            )
            del committed[request_id][args.history_blocks :]

    print(
        json.dumps(
            {
                "legacy_pickle_bytes": len(legacy_bytes),
                "delta_pickle_bytes": len(delta_bytes),
                "legacy_encode_us_p50": median_us(
                    lambda: pickle.dumps(
                        legacy,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ),
                    args.iterations,
                ),
                "delta_encode_us_p50": median_us(
                    lambda: pickle.dumps(
                        delta,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ),
                    args.iterations,
                ),
                "legacy_decode_us_p50": median_us(
                    lambda: pickle.loads(legacy_bytes),
                    args.iterations,
                ),
                "delta_decode_us_p50": median_us(
                    lambda: pickle.loads(delta_bytes),
                    args.iterations,
                ),
                "worker_apply_us_p50": median_us(
                    apply_delta,
                    args.iterations,
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
