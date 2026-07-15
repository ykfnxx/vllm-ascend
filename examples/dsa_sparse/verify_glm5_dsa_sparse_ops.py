#!/usr/bin/env python3
"""Send a long request and verify the DSA lookup/maintain operator logs.

Start the server with its output redirected to a log file before running this
script, for example:

    ./examples/dsa_sparse/serve_glm5_dsa_sparse.sh /path/to/GLM-5-w4a8 \
        2>&1 | tee /tmp/glm5-dsa.log

    python examples/dsa_sparse/verify_glm5_dsa_sparse_ops.py \
        --server-log /tmp/glm5-dsa.log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DSA_SPARSE_THRESHOLD = 10 * 1024 + 128
DEFAULT_TARGET_PROMPT_TOKENS = 10_600
DEFAULT_MAX_TOKENS = 4
PROMPT_FRAGMENT = "The cache lookup validation context contains deterministic tokens. "
OPERATOR_LOG_MARKERS = (
    "DSA sparse invoking asu_hbm_index_lookup",
    "DSA sparse completed asu_hbm_index_lookup",
    "DSA sparse invoking asu_hbm_index_maintain_aicpu",
    "DSA sparse completed asu_hbm_index_maintain_aicpu",
)


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
            f"POST {url} returned HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"POST {url} failed: {error.reason}") from error

    try:
        result = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"POST {url} returned a non-JSON response: {body[:500]}") from error
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
) -> dict[str, Any]:
    result = _post_json(
        f"{base_url}/tokenize",
        {"model": model, "prompt": prompt},
        api_key=api_key,
        timeout=timeout,
    )
    if not isinstance(result.get("count"), int):
        raise RuntimeError(f"/tokenize response has no integer count: {result}")
    return result


def _build_long_prompt(
    base_url: str,
    model: str,
    target_tokens: int,
    *,
    api_key: str | None,
    timeout: float,
) -> tuple[str, int, int]:
    unique_prefix = f"DSA lookup operator validation request {uuid.uuid4()}. "
    repeats = max(1, target_tokens // 8)

    for _ in range(6):
        prompt = unique_prefix + PROMPT_FRAGMENT * repeats
        tokenized = _tokenize(
            base_url,
            model,
            prompt,
            api_key=api_key,
            timeout=timeout,
        )
        prompt_tokens = int(tokenized["count"])
        max_model_len = int(tokenized.get("max_model_len", 0))
        if prompt_tokens >= target_tokens:
            return prompt, prompt_tokens, max_model_len
        repeats = max(
            repeats + 1,
            math.ceil(repeats * target_tokens / max(prompt_tokens, 1) * 1.02),
        )

    raise RuntimeError(
        f"failed to construct a prompt with at least {target_tokens} tokens")


def _read_log_from(log_path: Path, offset: int) -> str:
    with log_path.open("rb") as log_file:
        log_file.seek(offset)
        return log_file.read().decode("utf-8", errors="replace")


def _wait_for_operator_logs(
    log_path: Path,
    offset: int,
    wait_seconds: float,
) -> list[str]:
    deadline = time.monotonic() + wait_seconds
    missing = list(OPERATOR_LOG_MARKERS)
    while True:
        new_logs = _read_log_from(log_path, offset)
        missing = [marker for marker in OPERATOR_LOG_MARKERS if marker not in new_logs]
        if not missing or time.monotonic() >= deadline:
            return missing
        time.sleep(0.5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send a prompt above the DSA sparse threshold and verify the "
            "lookup/maintain operator logs."
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
        "--server-log",
        type=Path,
        required=True,
        help="file receiving the vLLM server stdout and stderr",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=DEFAULT_TARGET_PROMPT_TOKENS,
        help="minimum tokenized prompt length (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="number of completion tokens to request (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        help="Bearer token when the vLLM server requires authentication",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="HTTP timeout in seconds for the long request (default: %(default)s)",
    )
    parser.add_argument(
        "--log-wait-seconds",
        type=float,
        default=10.0,
        help="time to wait for server logs to flush (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.prompt_tokens <= DSA_SPARSE_THRESHOLD:
        print(
            "FAIL: --prompt-tokens must be greater than the DSA threshold "
            f"{DSA_SPARSE_THRESHOLD}",
            file=sys.stderr,
        )
        return 2
    if args.max_tokens < 2:
        print(
            "FAIL: --max-tokens must be at least 2 so a decode forward runs",
            file=sys.stderr,
        )
        return 2
    if not args.server_log.is_file():
        print(
            f"FAIL: server log does not exist: {args.server_log}",
            file=sys.stderr,
        )
        return 2
    existing_logs = _read_log_from(args.server_log, 0)
    existing_markers = [
        marker for marker in OPERATOR_LOG_MARKERS if marker in existing_logs
    ]
    if existing_markers:
        print(
            "FAIL: server log already contains DSA operator markers. Restart "
            "the service with a new log file before running this test because "
            "the framework logs each marker only once per process.",
            file=sys.stderr,
        )
        return 2

    base_url = args.base_url.rstrip("/")
    print(
        f"Building a prompt with at least {args.prompt_tokens} tokens "
        f"using {base_url}/tokenize ..."
    )
    try:
        prompt, prompt_tokens, max_model_len = _build_long_prompt(
            base_url,
            args.model,
            args.prompt_tokens,
            api_key=args.api_key,
            timeout=args.request_timeout,
        )
        if max_model_len and prompt_tokens + args.max_tokens > max_model_len:
            raise RuntimeError(
                "request exceeds the server model limit: "
                f"prompt_tokens={prompt_tokens}, max_tokens={args.max_tokens}, "
                f"max_model_len={max_model_len}"
            )

        log_offset = args.server_log.stat().st_size
        print(
            f"Sending completion request: prompt_tokens={prompt_tokens}, "
            f"max_tokens={args.max_tokens}, ignore_eos=true"
        )
        response = _post_json(
            f"{base_url}/v1/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": False,
            },
            api_key=args.api_key,
            timeout=args.request_timeout,
        )
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    usage = response.get("usage")
    if not isinstance(usage, dict):
        print(f"FAIL: completion response has no usage object: {response}", file=sys.stderr)
        return 1
    actual_prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(actual_prompt_tokens, int) or actual_prompt_tokens <= DSA_SPARSE_THRESHOLD:
        print(
            "FAIL: completion did not contain enough prompt tokens: "
            f"actual={actual_prompt_tokens}, threshold={DSA_SPARSE_THRESHOLD}",
            file=sys.stderr,
        )
        return 1
    if not isinstance(completion_tokens, int) or completion_tokens < 2:
        print(
            "FAIL: completion produced fewer than two tokens: "
            f"completion_tokens={completion_tokens}",
            file=sys.stderr,
        )
        return 1

    missing_markers = _wait_for_operator_logs(
        args.server_log,
        log_offset,
        args.log_wait_seconds,
    )
    if missing_markers:
        print("FAIL: request succeeded but new operator logs are missing:", file=sys.stderr)
        for marker in missing_markers:
            print(f"  - {marker}", file=sys.stderr)
        print(
            "Check that this is a freshly started DSA-enabled server. The "
            "framework prints each operator marker only once per process.",
            file=sys.stderr,
        )
        return 1

    print(
        "PASS: long completion succeeded and lookup/maintain invocation and "
        "completion logs were observed."
    )
    print(
        f"usage: prompt_tokens={actual_prompt_tokens}, "
        f"completion_tokens={completion_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
