# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post_json(
    url: str,
    payload: dict,
    request_id: str,
    timeout: float,
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def post_control(url: str, timeout: float) -> None:
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout):
        pass


def prepare_decode_requests(
    *,
    phase: str,
    prefill_url: str,
    model: str,
    batch_size: int,
    input_length: int,
    output_length: int,
    token_id: int,
    timeout: float,
) -> list[tuple[str, dict]]:
    prompt = [token_id] * input_length
    run_id = uuid.uuid4().hex

    def prefill(index: int) -> tuple[str, dict]:
        request_id = f"dsa-profile-{phase}-{run_id}-{index}"
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "min_tokens": 1,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": False,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        }
        response = post_json(prefill_url, payload, request_id, timeout)
        decode_payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": output_length,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": False,
            "kv_transfer_params": response["kv_transfer_params"],
        }
        return request_id, decode_payload

    print(
        f"{phase}: Prefill 1/{batch_size} populating the prefix cache...",
        flush=True,
    )
    start = time.perf_counter()
    prepared = [prefill(0)]
    print(
        f"{phase}: first Prefill completed in {time.perf_counter() - start:.1f}s",
        flush=True,
    )
    if batch_size > 1:
        print(
            f"{phase}: Prefill 2..{batch_size} reusing the cached prefix...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=batch_size - 1) as executor:
            prepared.extend(executor.map(prefill, range(1, batch_size)))
    return prepared


def run_decode_batch(
    decode_url: str,
    requests: list[tuple[str, dict]],
    timeout: float,
) -> list[dict]:
    barrier = threading.Barrier(len(requests))

    def decode(request: tuple[str, dict]) -> dict:
        request_id, payload = request
        barrier.wait()
        return post_json(decode_url, payload, request_id, timeout)

    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        return list(executor.map(decode, requests))


def profile(args: argparse.Namespace) -> None:
    prefill_url = f"http://127.0.0.1:{args.prefill_port}/v1/completions"
    decode_url = f"http://127.0.0.1:{args.decode_port}/v1/completions"
    profile_url = f"http://127.0.0.1:{args.decode_port}"

    print(
        f"Warming Decode Graph with input={args.warmup_input_length} "
        f"batch={args.batch_size} outside the profiling window...",
        flush=True,
    )
    warmup = prepare_decode_requests(
        phase="warmup",
        prefill_url=prefill_url,
        model=args.model,
        batch_size=args.batch_size,
        input_length=args.warmup_input_length,
        output_length=args.output_length,
        token_id=args.token_id,
        timeout=args.timeout,
    )
    run_decode_batch(decode_url, warmup, args.timeout)

    print(
        f"Preparing {args.batch_size} Prefill handoffs with "
        f"input={args.input_length}...",
        flush=True,
    )
    measured = prepare_decode_requests(
        phase="measured",
        prefill_url=prefill_url,
        model=args.model,
        batch_size=args.batch_size,
        input_length=args.input_length,
        output_length=args.output_length,
        token_id=args.token_id,
        timeout=args.timeout,
    )

    print("Starting Decode-only profile...", flush=True)
    post_control(f"{profile_url}/start_profile", args.timeout)
    try:
        responses = run_decode_batch(decode_url, measured, args.timeout)
    finally:
        post_control(f"{profile_url}/stop_profile", args.timeout)

    args.response_path.write_text(
        json.dumps(responses, separators=(",", ":")),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefill-port", type=int, required=True)
    parser.add_argument("--decode-port", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-input-length", type=int, required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--token-id", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--response-path", type=Path, required=True)
    profile(parser.parse_args())


if __name__ == "__main__":
    main()
