# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Validate exact DSA Sparse P/D shared-memory payload delivery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PD_LOG_PREFIX = "DSA_SPARSE_PD "
PROBE_LOG_PREFIX = "DSA_SPARSE_PROBE "


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _read_events(
    log_path: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            _, marker, payload = line.partition(prefix)
            if not marker:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as error:
                _fail(
                    f"Malformed event at {log_path}:{line_number}: {error}"
                )
            if not isinstance(event, dict):
                _fail(
                    f"Event must be a JSON object at "
                    f"{log_path}:{line_number}"
                )
            events.append(event)
    return events


def _only_event(
    events: list[dict[str, Any]],
    event_name: str,
    source: str,
) -> dict[str, Any]:
    matches = [
        event for event in events if event.get("event") == event_name
    ]
    if len(matches) != 1:
        _fail(
            f"Expected exactly one {event_name} event in {source}, "
            f"got {len(matches)}"
        )
    return matches[0]


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string, got {value!r}")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer, got {value!r}")
    return value


def _payloads_from_handoff(
    handoff: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    by_rank = handoff.get("shared_memory_payloads_by_rank")
    if not isinstance(by_rank, dict) or not by_rank:
        _fail("Handoff has no shared-memory payload ranks")
    payloads: dict[str, dict[str, Any]] = {}
    for rank, layers in by_rank.items():
        if not isinstance(layers, dict) or not layers:
            _fail(f"Handoff rank {rank!r} has no shared-memory payloads")
        for layer_name, payload in layers.items():
            if not isinstance(layer_name, str) or not isinstance(payload, dict):
                _fail(
                    f"Invalid shared-memory payload at rank {rank!r}, "
                    f"layer {layer_name!r}"
                )
            object_name = _require_string(
                payload.get("name"),
                f"{rank}.{layer_name}.name",
            )
            if object_name in payloads:
                _fail(f"Duplicate shared-memory object: {object_name!r}")
            content_sha256 = _require_string(
                payload.get("content_sha256"),
                f"{object_name}.content_sha256",
            )
            if len(content_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in content_sha256
            ):
                _fail(
                    f"{object_name}.content_sha256 is not a SHA-256 digest"
                )
            cache_planes = payload.get("cache_planes")
            tail_planes = payload.get("tail_planes")
            if not isinstance(cache_planes, list) or not cache_planes:
                _fail(f"{object_name} has no cache planes")
            if not isinstance(tail_planes, list):
                _fail(f"{object_name} has invalid tail planes")
            payloads[object_name] = {
                **payload,
                "rank": str(rank),
                "layer": layer_name,
            }
    return payloads


def _events_by_object(
    events: list[dict[str, Any]],
    event_name: str,
    source: str,
) -> dict[str, dict[str, Any]]:
    by_object: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") != event_name:
            continue
        object_name = _require_string(
            event.get("object_name"),
            f"{event_name}.object_name",
        )
        if object_name in by_object:
            _fail(
                f"{source} contains duplicate {event_name} event for "
                f"{object_name!r}"
            )
        by_object[object_name] = event
    return by_object


def _validate_payload_events(
    *,
    expected_payloads: dict[str, dict[str, Any]],
    actual_events: dict[str, dict[str, Any]],
    event_name: str,
) -> None:
    if set(actual_events) != set(expected_payloads):
        _fail(
            f"{event_name} objects mismatch: expected="
            f"{sorted(expected_payloads)}, actual={sorted(actual_events)}"
        )
    for object_name, payload in expected_payloads.items():
        event = actual_events[object_name]
        expected_fields = {
            "cache_kind": payload.get("cache_kind"),
            "content_sha256": payload.get("content_sha256"),
            "payload_bytes": payload.get("size"),
        }
        for field, expected in expected_fields.items():
            if event.get(field) != expected:
                _fail(
                    f"{event_name}.{object_name}.{field} mismatch: "
                    f"expected={expected!r}, actual={event.get(field)!r}"
                )


def validate(
    *,
    prefill_log: Path,
    decode_log: Path,
    shared_memory_root: Path,
    require_mtp: bool,
) -> dict[str, int]:
    prefill_pd_events = _read_events(prefill_log, PD_LOG_PREFIX)
    decode_pd_events = _read_events(decode_log, PD_LOG_PREFIX)
    send = _only_event(
        prefill_pd_events,
        "handoff_send",
        str(prefill_log),
    )
    receive = _only_event(
        decode_pd_events,
        "handoff_receive",
        str(decode_log),
    )
    if send.get("handoff_sha256") != receive.get("handoff_sha256"):
        _fail("P and D handoff SHA-256 values do not match")
    if send.get("handoff") != receive.get("handoff"):
        _fail("P and D serialized handoffs do not match")
    handoff = send.get("handoff")
    if not isinstance(handoff, dict):
        _fail("Prefill handoff payload is not a dictionary")

    expected_payloads = _payloads_from_handoff(handoff)
    cache_kind_counts = Counter(
        str(payload.get("cache_kind"))
        for payload in expected_payloads.values()
    )
    if cache_kind_counts["indexer"] <= 0:
        _fail("Handoff contains no Indexer shared-memory payload")
    if require_mtp and cache_kind_counts["mtp_draft"] <= 0:
        _fail("MTP was requested but the handoff contains no MTP draft payload")
    if not require_mtp and cache_kind_counts["mtp_draft"]:
        _fail("Handoff unexpectedly contains an MTP draft payload")

    stored_token_count = _require_positive_int(
        handoff.get("stored_token_count"),
        "handoff.stored_token_count",
    )
    block_size = _require_positive_int(
        handoff.get("block_size"),
        "handoff.block_size",
    )
    partial_tail = stored_token_count % block_size != 0
    tail_payload_count = 0
    for object_name, payload in expected_payloads.items():
        tail_planes = payload["tail_planes"]
        cache_kind = payload.get("cache_kind")
        if tail_planes:
            tail_payload_count += 1
            if cache_kind != "indexer":
                _fail(f"Non-Indexer payload {object_name!r} contains a tail")
        if partial_tail and cache_kind == "indexer" and not tail_planes:
            _fail(f"Indexer payload {object_name!r} is missing its partial tail")

    prefill_probe_events = _read_events(prefill_log, PROBE_LOG_PREFIX)
    decode_probe_events = _read_events(decode_log, PROBE_LOG_PREFIX)
    published = _events_by_object(
        prefill_probe_events,
        "shared_memory_publish",
        str(prefill_log),
    )
    verified = _events_by_object(
        decode_probe_events,
        "shared_memory_content_verified",
        str(decode_log),
    )
    consumed = _events_by_object(
        decode_probe_events,
        "shared_memory_consume",
        str(decode_log),
    )
    _validate_payload_events(
        expected_payloads=expected_payloads,
        actual_events=published,
        event_name="shared_memory_publish",
    )
    _validate_payload_events(
        expected_payloads=expected_payloads,
        actual_events=verified,
        event_name="shared_memory_content_verified",
    )
    _validate_payload_events(
        expected_payloads=expected_payloads,
        actual_events=consumed,
        event_name="shared_memory_consume",
    )

    leaked_objects = sorted(
        object_name
        for object_name in expected_payloads
        if (shared_memory_root / object_name).exists()
    )
    if leaked_objects:
        _fail(
            "Consumed shared-memory objects still exist: "
            f"{leaked_objects}"
        )

    return {
        "payload_count": len(expected_payloads),
        "indexer_payload_count": cache_kind_counts["indexer"],
        "mtp_payload_count": cache_kind_counts["mtp_draft"],
        "tail_payload_count": tail_payload_count,
        "verified_payload_count": len(verified),
        "released_payload_count": len(consumed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate bit-exact P to shared memory to Decode-cache delivery "
            "for DSA Sparse Indexer, partial Main Tail, and MTP draft payloads."
        )
    )
    parser.add_argument("--prefill-log", type=Path, required=True)
    parser.add_argument("--decode-log", type=Path, required=True)
    parser.add_argument(
        "--shared-memory-root",
        type=Path,
        default=Path("/dev/shm"),
    )
    parser.add_argument("--require-mtp", action="store_true")
    args = parser.parse_args()
    summary = validate(
        prefill_log=args.prefill_log,
        decode_log=args.decode_log,
        shared_memory_root=args.shared_memory_root,
        require_mtp=args.require_mtp,
    )
    print(
        "PASS: DSA Sparse shared-memory P/D content verified: "
        + json.dumps(summary, sort_keys=True)
    )


if __name__ == "__main__":
    main()
