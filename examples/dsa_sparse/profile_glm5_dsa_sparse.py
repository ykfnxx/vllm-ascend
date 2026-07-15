#!/usr/bin/env python3
"""Profile long-context GLM-5 serving with controlled request concurrency.

The workload is compatible with both the DSA sparse branch and an unmodified
vLLM Ascend v0.18.0 server. DSA sparse execution is selected by the server;
this client guarantees the request-side conditions needed to enter that path.

Example:

    python3 examples/dsa_sparse/profile_glm5_dsa_sparse.py \
        --batch-sizes 1 2 4 8 \
        --profile \
        --server-log /tmp/glm5-dsa.log \
        --output-json /tmp/glm5-dsa-profile.json

The server must be started with ``--profiler-config`` when ``--profile`` is
used. The generated NPU traces are written by the server, not this client.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DSA_QUERY_TOKENS = 2 * 1024
DSA_RESIDENT_TOKENS = 8 * 1024
DSA_BLOCK_SIZE = 128
DSA_SPARSE_THRESHOLD = DSA_QUERY_TOKENS + DSA_RESIDENT_TOKENS + DSA_BLOCK_SIZE

DEFAULT_TARGET_PROMPT_TOKENS = 10_600
# Leave enough decode steps for long prefills in the same wave to converge into
# one sparse decode batch when the server uses 4096-token chunked prefill.
DEFAULT_MAX_TOKENS = 32
PROMPT_FRAGMENT = (
    "The DSA sparse profiling context contains deterministic token data for "
    "cache lookup and materialization. "
)
OPERATOR_LOG_MARKERS = (
    "DSA sparse invoking asu_hbm_index_lookup",
    "DSA sparse completed asu_hbm_index_lookup",
    "DSA sparse invoking asu_hbm_index_maintain_aicpu",
    "DSA sparse completed asu_hbm_index_maintain_aicpu",
)


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    ttft_seconds: float


@dataclass(frozen=True)
class WaveResult:
    round: int
    wall_seconds: float
    requests: list[RequestResult]


def _request(
    url: str,
    *,
    payload: dict[str, Any] | None,
    api_key: str | None,
    timeout: float,
) -> bytes:
    headers: dict[str, str] = {}
    data = b""
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"POST {url} returned HTTP {error.code}: {body}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"POST {url} failed: {error.reason}") from error


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    body = _request(
        url,
        payload=payload,
        api_key=api_key,
        timeout=timeout,
    ).decode("utf-8")
    try:
        result = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"POST {url} returned a non-JSON response: {body[:500]}"
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
        raise RuntimeError(f"/tokenize response has no integer count: {result}")
    max_model_len = result.get("max_model_len", 0)
    if not isinstance(max_model_len, int):
        max_model_len = 0
    return count, max_model_len


def _build_long_prompt(
    base_url: str,
    model: str,
    target_tokens: int,
    *,
    api_key: str | None,
    timeout: float,
) -> tuple[str, int, int]:
    prefix = "DSA sparse controlled-concurrency profiling context. "
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


def _stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
    *,
    api_key: str | None,
    timeout: float,
    ready: queue.Queue[None],
    start_event: threading.Event,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Prevent prefix-cache sharing between concurrent requests and cases.
        "cache_salt": request_id,
        "request_id": request_id,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    ready.put(None)
    start_event.wait()
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] | None = None

    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"request {request_id} received invalid SSE JSON: {data[:500]}"
                    ) from error
                if not isinstance(event, dict):
                    continue
                if "error" in event:
                    raise RuntimeError(
                        f"request {request_id} returned an error: {event['error']}"
                    )
                choices = event.get("choices")
                if first_token_at is None and isinstance(choices, list) and choices:
                    first_token_at = time.perf_counter()
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = event_usage
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"request {request_id} returned HTTP {error.code}: {body}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"request {request_id} failed: {error.reason}"
        ) from error

    finished = time.perf_counter()
    if usage is None:
        raise RuntimeError(f"request {request_id} returned no usage information")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        raise RuntimeError(f"request {request_id} returned invalid usage: {usage}")
    if prompt_tokens <= DSA_SPARSE_THRESHOLD:
        raise RuntimeError(
            f"request {request_id} prompt has {prompt_tokens} tokens; DSA sparse "
            f"requires more than {DSA_SPARSE_THRESHOLD}"
        )
    if completion_tokens < 2:
        raise RuntimeError(
            f"request {request_id} produced only {completion_tokens} tokens; "
            "at least two are required to execute sparse decode"
        )
    if first_token_at is None:
        raise RuntimeError(f"request {request_id} returned no streamed token event")

    return RequestResult(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_seconds=finished - started,
        ttft_seconds=first_token_at - started,
    )


def _run_wave(
    *,
    batch_size: int,
    round_index: int,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None,
    timeout: float,
) -> WaveResult:
    ready: queue.Queue[None] = queue.Queue()
    start_event = threading.Event()
    run_id = uuid.uuid4().hex
    futures: list[Future[RequestResult]] = []

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for request_index in range(batch_size):
            request_id = f"{run_id}-{request_index}"
            futures.append(
                executor.submit(
                    _stream_completion,
                    base_url,
                    model,
                    prompt,
                    max_tokens,
                    request_id,
                    api_key=api_key,
                    timeout=timeout,
                    ready=ready,
                    start_event=start_event,
                )
            )

        for _ in range(batch_size):
            ready.get(timeout=timeout)
        wave_started = time.perf_counter()
        start_event.set()
        results = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - wave_started

    return WaveResult(
        round=round_index,
        wall_seconds=wall_seconds,
        requests=results,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency_summary(values: list[float]) -> dict[str, float]:
    values_ms = [value * 1000 for value in values]
    return {
        "mean_ms": statistics.fmean(values_ms),
        "p50_ms": _percentile(values_ms, 50),
        "p90_ms": _percentile(values_ms, 90),
        "p99_ms": _percentile(values_ms, 99),
    }


def _summarize_case(
    batch_size: int,
    waves: list[WaveResult],
) -> dict[str, Any]:
    requests = [request for wave in waves for request in wave.requests]
    wall_seconds = sum(wave.wall_seconds for wave in waves)
    completion_tokens = sum(request.completion_tokens for request in requests)
    total_tokens = sum(
        request.prompt_tokens + request.completion_tokens for request in requests
    )
    return {
        "batch_size": batch_size,
        "rounds": len(waves),
        "request_count": len(requests),
        "wall_seconds": wall_seconds,
        "request_throughput_per_second": len(requests) / wall_seconds,
        "output_token_throughput_per_second": completion_tokens / wall_seconds,
        "total_token_throughput_per_second": total_tokens / wall_seconds,
        "ttft": _latency_summary([request.ttft_seconds for request in requests]),
        "latency": _latency_summary(
            [request.latency_seconds for request in requests]
        ),
        "waves": [
            {
                "round": wave.round,
                "wall_seconds": wave.wall_seconds,
                "requests": [asdict(request) for request in wave.requests],
            }
            for wave in waves
        ],
    }


def _profile_control(
    base_url: str,
    action: str,
    *,
    api_key: str | None,
    timeout: float,
) -> None:
    _request(
        f"{base_url}/{action}_profile",
        payload=None,
        api_key=api_key,
        timeout=timeout,
    )


def _verify_operator_logs(path: Path) -> dict[str, bool]:
    logs = path.read_text(encoding="utf-8", errors="replace")
    return {marker: marker in logs for marker in OPERATOR_LOG_MARKERS}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile GLM-5 with synchronized long-context request batches that "
            "enter DSA sparse decode."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8077",
        help="vLLM server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="glm-5",
        help="served model name (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="concurrent requests in each profiled case (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=DEFAULT_TARGET_PROMPT_TOKENS,
        help="minimum prompt token count (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="completion tokens per request (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="unprofiled warmup waves per batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="profiled/measured waves per batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="call the server /start_profile and /stop_profile endpoints",
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        help="DSA server log; when set, lookup/maintain markers are required",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="write measurements and run parameters to this JSON file",
    )
    parser.add_argument(
        "--label",
        default="dsa-sparse",
        help="run label stored in output JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        help="Bearer token when the vLLM server requires authentication",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.prompt_tokens <= DSA_SPARSE_THRESHOLD:
        raise ValueError(
            f"--prompt-tokens must be greater than {DSA_SPARSE_THRESHOLD}"
        )
    if args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2")
    if args.warmup_rounds < 0:
        raise ValueError("--warmup-rounds cannot be negative")
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if not args.batch_sizes or any(size < 1 for size in args.batch_sizes):
        raise ValueError("--batch-sizes values must be positive")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("--batch-sizes values must be unique")
    if args.server_log is not None and not args.server_log.is_file():
        raise ValueError(f"server log does not exist: {args.server_log}")


def main() -> int:
    args = _parse_args()
    try:
        _validate_args(args)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    try:
        print(
            f"Building prompt with at least {args.prompt_tokens} tokens via "
            f"{base_url}/tokenize ..."
        )
        prompt, prompt_tokens, max_model_len = _build_long_prompt(
            base_url,
            args.model,
            args.prompt_tokens,
            api_key=args.api_key,
            timeout=args.request_timeout,
        )
        if max_model_len and prompt_tokens + args.max_tokens > max_model_len:
            raise RuntimeError(
                "request exceeds server model limit: "
                f"prompt_tokens={prompt_tokens}, max_tokens={args.max_tokens}, "
                f"max_model_len={max_model_len}"
            )

        cases: list[dict[str, Any]] = []
        for batch_size in args.batch_sizes:
            print(
                f"batch_size={batch_size}: warmup_rounds={args.warmup_rounds}, "
                f"measured_rounds={args.rounds}, max_tokens={args.max_tokens}"
            )
            for warmup_round in range(args.warmup_rounds):
                _run_wave(
                    batch_size=batch_size,
                    round_index=warmup_round,
                    base_url=base_url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    api_key=args.api_key,
                    timeout=args.request_timeout,
                )

            profile_started = False
            try:
                if args.profile:
                    _profile_control(
                        base_url,
                        "start",
                        api_key=args.api_key,
                        timeout=args.request_timeout,
                    )
                    profile_started = True
                    print(f"batch_size={batch_size}: profiler started")

                waves = [
                    _run_wave(
                        batch_size=batch_size,
                        round_index=round_index,
                        base_url=base_url,
                        model=args.model,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        api_key=args.api_key,
                        timeout=args.request_timeout,
                    )
                    for round_index in range(args.rounds)
                ]
            finally:
                if profile_started:
                    _profile_control(
                        base_url,
                        "stop",
                        api_key=args.api_key,
                        timeout=args.request_timeout,
                    )
                    print(f"batch_size={batch_size}: profiler stopped")

            case = _summarize_case(batch_size, waves)
            cases.append(case)
            print(
                f"batch_size={batch_size}: "
                f"requests/s={case['request_throughput_per_second']:.4f}, "
                f"output_tokens/s={case['output_token_throughput_per_second']:.4f}, "
                f"TTFT_p50={case['ttft']['p50_ms']:.3f} ms, "
                f"latency_p50={case['latency']['p50_ms']:.3f} ms"
            )

        marker_status: dict[str, bool] | None = None
        if args.server_log is not None:
            marker_status = _verify_operator_logs(args.server_log)
            missing = [
                marker for marker, present in marker_status.items() if not present
            ]
            if missing:
                raise RuntimeError(
                    "DSA operator log markers are missing: " + ", ".join(missing)
                )
            print("DSA lookup/maintain invocation and completion logs are present")

        output = {
            "label": args.label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "base_url": base_url,
            "model": args.model,
            "profile_enabled": args.profile,
            "prompt_tokens": prompt_tokens,
            "target_prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "warmup_rounds": args.warmup_rounds,
            "measured_rounds": args.rounds,
            "dsa_sparse_threshold": DSA_SPARSE_THRESHOLD,
            "operator_log_markers": marker_status,
            "cases": cases,
        }
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(output, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            print(f"Results written to {args.output_json}")
    except (RuntimeError, OSError, queue.Empty) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
