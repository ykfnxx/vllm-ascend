#!/usr/bin/env python3
"""Start a mock-backend DSA server and test concurrent request batches."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DSA_BLOCK_SIZE = 128
DSA_RESIDENT_TOKENS = 8 * 1024
DSA_QUERY_TOKENS = 2 * 1024
DSA_TOTAL_SLOTS = DSA_RESIDENT_TOKENS + DSA_QUERY_TOKENS
DSA_SPARSE_THRESHOLD = DSA_TOTAL_SLOTS + DSA_BLOCK_SIZE
DSA_BLOCKS_PER_REQUEST = math.ceil(DSA_SPARSE_THRESHOLD / DSA_BLOCK_SIZE)
PROMPT_FRAGMENT = (
    "The DSA sparse batch test contains deterministic token data for cache "
    "lookup and mock backend loading. "
)
SHARED_CACHE_SALT = "dsa-mock-batch-test"
REQUIRED_LOG_MARKERS = (
    "DSA mock KV backend accepted block puts without storage",
    "DSA mock KV backend wrote lookup misses directly to resident HBM",
    "DSA sparse invoking asu_hbm_index_lookup",
    "DSA sparse completed asu_hbm_index_lookup",
    "DSA sparse invoking asu_hbm_index_maintain_aicpu",
    "DSA sparse completed asu_hbm_index_maintain_aicpu",
)


def _parse_args() -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Start vLLM with the mock DSA KV backend, run concurrent long-"
            "context batches, verify operator logs, and stop the server."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="concurrent request counts tested by the same server",
    )
    parser.add_argument("--prompt-tokens", type=int, default=10_600)
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="output tokens per request; defaults to max(32, 4 * max batch)",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--served-model-name", default="glm-5")
    parser.add_argument("--tensor-parallel-size", type=int, default=16)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=131_072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        help="explicitly override vLLM's automatic NPU KV block count",
    )
    parser.add_argument("--server-start-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--api-key")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"dsa-batch-test-results/{timestamp}"),
    )
    parser.add_argument(
        "vllm_args",
        nargs=argparse.REMAINDER,
        help="extra vllm serve arguments after --",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model_path.is_dir():
        raise ValueError(f"model directory does not exist: {args.model_path}")
    if not args.batch_sizes or any(size <= 0 for size in args.batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("--batch-sizes values must be unique")
    if args.prompt_tokens <= DSA_SPARSE_THRESHOLD:
        raise ValueError(
            f"--prompt-tokens must be greater than {DSA_SPARSE_THRESHOLD}"
        )
    if args.max_tokens is not None and args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if (
        args.num_gpu_blocks_override is not None
        and args.num_gpu_blocks_override <= 0
    ):
        raise ValueError("--num-gpu-blocks-override must be positive")
    if args.num_gpu_blocks_override is not None:
        min_gpu_blocks = max(args.batch_sizes) * DSA_BLOCKS_PER_REQUEST + 1
        if args.num_gpu_blocks_override < min_gpu_blocks:
            recommended_gpu_blocks = math.ceil(
                min_gpu_blocks / DSA_BLOCK_SIZE
            ) * DSA_BLOCK_SIZE
            raise ValueError(
                "--num-gpu-blocks-override="
                f"{args.num_gpu_blocks_override} is too small for max batch "
                f"size {max(args.batch_sizes)}; minimum={min_gpu_blocks}, "
                f"recommended={recommended_gpu_blocks}"
            )
    if args.server_start_timeout <= 0 or args.request_timeout <= 0:
        raise ValueError("timeouts must be positive")
    if shutil.which("vllm") is None:
        raise ValueError("vllm executable is not available in PATH")


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"POST {url} returned HTTP {error.code}: {body}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"POST {url} failed: {error.reason}") from error

    try:
        result = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"POST {url} returned non-JSON data: {body[:500]}"
        ) from error
    if not isinstance(result, dict):
        raise RuntimeError(f"POST {url} returned unexpected JSON: {result!r}")
    return result


def _tokenize(
    base_url: str,
    model: str,
    prompt: str,
    *,
    api_key: str | None,
    timeout: float,
) -> tuple[int, int]:
    result = _post_json(
        f"{base_url}/tokenize",
        {"model": model, "prompt": prompt},
        api_key=api_key,
        timeout=timeout,
    )
    count = result.get("count")
    if not isinstance(count, int):
        raise RuntimeError(f"/tokenize returned no integer count: {result}")
    max_model_len = result.get("max_model_len", 0)
    return count, max_model_len if isinstance(max_model_len, int) else 0


def _build_long_prompt(
    base_url: str,
    model: str,
    target_tokens: int,
    *,
    api_key: str | None,
    timeout: float,
) -> tuple[str, int, int]:
    prefix = "DSA sparse mock-backend concurrent batch test. "
    repeats = max(1, target_tokens // 12)
    for _ in range(8):
        prompt = prefix + PROMPT_FRAGMENT * repeats
        prompt_tokens, max_model_len = _tokenize(
            base_url,
            model,
            prompt,
            api_key=api_key,
            timeout=timeout,
        )
        if prompt_tokens >= target_tokens:
            return prompt, prompt_tokens, max_model_len
        repeats = max(
            repeats + 1,
            math.ceil(repeats * target_tokens / max(prompt_tokens, 1) * 1.01),
        )
    raise RuntimeError(
        f"failed to construct a prompt with at least {target_tokens} tokens"
    )


def _send_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
    cache_salt: str,
    api_key: str | None,
    timeout: float,
    ready: queue.Queue[None],
    start_event: threading.Event,
) -> dict[str, int | str]:
    ready.put(None)
    start_event.wait()
    result = _post_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "cache_salt": cache_salt,
            "request_id": request_id,
        },
        api_key=api_key,
        timeout=timeout,
    )
    if "error" in result:
        raise RuntimeError(f"request {request_id} returned: {result['error']}")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"request {request_id} returned no completion: {result}")
    usage = result.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(f"request {request_id} returned no usage: {result}")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        raise RuntimeError(f"request {request_id} returned invalid usage: {usage}")
    if prompt_tokens <= DSA_SPARSE_THRESHOLD or completion_tokens < 2:
        raise RuntimeError(
            f"request {request_id} did not execute sparse decode: "
            f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
        )
    return {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _warm_prefix_cache(
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float,
) -> None:
    request_id = f"prefix-warmup-{uuid.uuid4().hex}"
    result = _post_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "ignore_eos": True,
            "cache_salt": SHARED_CACHE_SALT,
            "request_id": request_id,
        },
        api_key=api_key,
        timeout=timeout,
    )
    choices = result.get("choices")
    usage = result.get("usage")
    if not isinstance(choices, list) or not choices or not isinstance(usage, dict):
        raise RuntimeError(f"prefix warmup returned an invalid response: {result}")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or prompt_tokens <= DSA_SPARSE_THRESHOLD
        or completion_tokens != 1
    ):
        raise RuntimeError(f"prefix warmup returned invalid usage: {usage}")


def _run_batch(
    *,
    batch_size: int,
    round_index: int,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None,
    timeout: float,
) -> list[dict[str, int | str]]:
    ready: queue.Queue[None] = queue.Queue()
    start_event = threading.Event()
    run_id = uuid.uuid4().hex
    futures: list[Future[dict[str, int | str]]] = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for request_index in range(batch_size):
            request_id = (
                f"batch{batch_size}-r{round_index}-{run_id}-{request_index}"
            )
            futures.append(
                executor.submit(
                    _send_completion,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    request_id=request_id,
                    cache_salt=SHARED_CACHE_SALT,
                    api_key=api_key,
                    timeout=timeout,
                    ready=ready,
                    start_event=start_event,
                )
            )
        for _ in range(batch_size):
            ready.get(timeout=timeout)
        start_event.set()
        return [future.result() for future in futures]


def _wait_for_server(
    process: subprocess.Popen[bytes],
    health_url: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"vllm serve exited before readiness with code {return_code}"
            )
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(2)
    raise RuntimeError(f"server was not ready within {timeout} seconds")


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _write_command(path: Path, command: list[str]) -> None:
    rendered = list(command)
    for index, argument in enumerate(rendered[:-1]):
        if argument == "--api-key":
            rendered[index + 1] = "<redacted>"
    path.write_text(shlex.join(rendered) + "\n", encoding="utf-8")


def _verify_logs(path: Path) -> dict[str, bool]:
    logs = path.read_text(encoding="utf-8", errors="replace")
    status = {marker: marker in logs for marker in REQUIRED_LOG_MARKERS}
    missing = [marker for marker, present in status.items() if not present]
    if missing:
        raise RuntimeError("required DSA log markers are missing: " + ", ".join(missing))
    return status


def main() -> int:
    args = _parse_args()
    try:
        _validate_args(args)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    max_batch_size = max(args.batch_sizes)
    max_tokens = args.max_tokens or max(32, 4 * max_batch_size)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = output_dir / "server.log"
    results_path = output_dir / "results.json"
    command_path = output_dir / "serve-command.txt"
    base_url = f"http://{args.host}:{args.port}"

    additional_config = json.dumps(
        {
            "fuse_muls_add": True,
            "multistream_overlap_shared_expert": True,
            "dsa_sparse_config": {
                "enabled": True,
                "kv_backend": "mock",
                "max_active_reqs": max_batch_size,
            },
        },
        separators=(",", ":"),
    )
    extra_args = list(args.vllm_args)
    if extra_args and extra_args[0] == "--":
        extra_args.pop(0)
    command = [
        "vllm",
        "serve",
        str(args.model_path.resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--enable-expert-parallel",
        "--seed",
        "1024",
        "--served-model-name",
        args.served_model_name,
        "--max-num-seqs",
        str(max_batch_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.95",
        "--quantization",
        "ascend",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--enforce-eager",
        "--no-async-scheduling",
        "--block-size",
        str(DSA_BLOCK_SIZE),
        "--additional-config",
        additional_config,
    ]
    if args.num_gpu_blocks_override is not None:
        command.extend(
            ["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)]
        )
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    command.extend(extra_args)
    _write_command(command_path, command)

    environment = os.environ.copy()
    environment.pop("VLLM_ASCEND_BALANCE_SCHEDULING", None)
    environment.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
    environment.pop("VLLM_ASCEND_ENABLE_FLASHCOMM", None)

    print(f"Output directory: {output_dir}")
    gpu_blocks = (
        "auto"
        if args.num_gpu_blocks_override is None
        else str(args.num_gpu_blocks_override)
    )
    print(
        f"Starting mock-backend server: max_batch={max_batch_size}, "
        f"gpu_blocks={gpu_blocks}, max_tokens={max_tokens}"
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        with server_log_path.open("wb") as server_log:
            process = subprocess.Popen(
                command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            _wait_for_server(
                process,
                f"{base_url}/health",
                args.server_start_timeout,
            )
            print("Server is ready; constructing the long prompt")
            prompt, prompt_tokens, server_max_model_len = _build_long_prompt(
                base_url,
                args.served_model_name,
                args.prompt_tokens,
                api_key=args.api_key,
                timeout=args.request_timeout,
            )
            if (
                server_max_model_len
                and prompt_tokens + DSA_BLOCK_SIZE + max_tokens
                > server_max_model_len
            ):
                raise RuntimeError(
                    "request exceeds server model limit: "
                    f"prompt={prompt_tokens}, output={max_tokens}, "
                    f"limit={server_max_model_len}"
                )

            print("Warming the shared long-prefix cache")
            _warm_prefix_cache(
                base_url=base_url,
                model=args.served_model_name,
                prompt=prompt,
                api_key=args.api_key,
                timeout=args.request_timeout,
            )

            cases: list[dict[str, Any]] = []
            for batch_size in args.batch_sizes:
                round_results = []
                for round_index in range(args.rounds):
                    print(
                        f"Testing batch_size={batch_size}, "
                        f"round={round_index + 1}/{args.rounds}"
                    )
                    requests = _run_batch(
                        batch_size=batch_size,
                        round_index=round_index,
                        base_url=base_url,
                        model=args.served_model_name,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        api_key=args.api_key,
                        timeout=args.request_timeout,
                    )
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"server exited with code {process.returncode} "
                            f"while testing batch_size={batch_size}"
                        )
                    round_results.append(requests)
                cases.append(
                    {
                        "batch_size": batch_size,
                        "rounds": round_results,
                        "status": "pass",
                    }
                )
                print(f"PASS: batch_size={batch_size}")

            server_log.flush()
            log_markers = _verify_logs(server_log_path)
            results_path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "model_path": str(args.model_path.resolve()),
                        "batch_sizes": args.batch_sizes,
                        "prompt_tokens": prompt_tokens,
                        "max_tokens": max_tokens,
                        "num_gpu_blocks_override": args.num_gpu_blocks_override,
                        "log_markers": log_markers,
                        "cases": cases,
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"PASS: all batch sizes completed; results={results_path}")
    except (OSError, RuntimeError, queue.Empty) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        print(f"Server log: {server_log_path}", file=sys.stderr)
        return 1
    finally:
        if process is not None:
            _stop_server(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
