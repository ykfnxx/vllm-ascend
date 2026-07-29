# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROBE_LOG_PREFIX = "DSA_SPARSE_PROBE "
CUSTOM_OP_NAMES = (
    "dsasparselookupupdate",
    "dsa_sparse_lookup_update",
    "aclnndsasparselookupupdate",
)
PROFILE_FILENAMES = {
    "kernel_details.csv",
    "operator_details.csv",
    "op_statistic.csv",
}


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _read_probe_events(decode_log: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with decode_log.open(encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            _, marker, payload = line.partition(PROBE_LOG_PREFIX)
            if not marker:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as error:
                _fail(
                    "Malformed DSA Sparse probe event at "
                    f"{decode_log}:{line_number}: {error}"
                )
            if not isinstance(event, dict):
                _fail(
                    "DSA Sparse probe event must be a JSON object at "
                    f"{decode_log}:{line_number}"
                )
            events.append(event)
    if not events:
        _fail(f"No DSA Sparse probe events found in {decode_log}")
    return events


def _events_named(
    events: list[dict[str, Any]],
    name: str,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("event") == name
    ]


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer, got {value!r}")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string, got {value!r}")
    return value


def _require_int_list(value: object, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in value
        )
    ):
        _fail(f"{field} must be a non-empty integer list, got {value!r}")
    return tuple(value)


def _read_completion_tokens(response_json: Path) -> int:
    with response_json.open(encoding="utf-8") as response_file:
        response = json.load(response_file)
    if not isinstance(response, dict):
        _fail("Response JSON must contain an object.")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        _fail("Response JSON does not contain usage metadata.")
    return _require_positive_int(
        usage.get("completion_tokens"),
        "usage.completion_tokens",
    )


def _profile_contains_custom_op(profile_dir: Path) -> bool:
    profile_files = [
        path
        for path in profile_dir.rglob("*")
        if path.is_file() and path.name in PROFILE_FILENAMES
    ]
    if not profile_files:
        _fail(
            "No analyzed operator or kernel CSV files found under "
            f"{profile_dir}"
        )
    for profile_file in profile_files:
        with profile_file.open(
            encoding="utf-8",
            errors="replace",
        ) as csv_file:
            for line in csv_file:
                lowered = line.lower()
                if any(
                    op_name in lowered
                    for op_name in CUSTOM_OP_NAMES
                ):
                    return True
    return False


def validate(
    *,
    decode_log: Path,
    response_json: Path,
    profile_dir: Path,
) -> dict[str, int]:
    events = _read_probe_events(decode_log)
    runtime_events = _events_named(events, "runtime_ready")
    if len(runtime_events) != 1:
        _fail(
            "Expected exactly one runtime_ready event, got "
            f"{len(runtime_events)}"
        )
    runtime = runtime_events[0]
    cohort_count = _require_positive_int(
        runtime.get("cohort_count"),
        "runtime_ready.cohort_count",
    )
    layer_count = _require_positive_int(
        runtime.get("layer_count"),
        "runtime_ready.layer_count",
    )
    index_topk = _require_positive_int(
        runtime.get("index_topk"),
        "runtime_ready.index_topk",
    )

    cohorts = runtime.get("cohorts")
    if not isinstance(cohorts, list) or len(cohorts) != cohort_count:
        _fail(
            "runtime_ready.cohorts does not match cohort_count: "
            f"{cohorts!r}"
        )
    cohort_layers: dict[str, tuple[str, ...]] = {}
    for cohort in cohorts:
        if not isinstance(cohort, dict):
            _fail(f"Invalid cohort descriptor: {cohort!r}")
        cohort_name = _require_string(
            cohort.get("name"),
            "cohort.name",
        )
        layers = cohort.get("layers")
        if (
            not isinstance(layers, list)
            or not layers
            or any(
                not isinstance(layer, str) or not layer
                for layer in layers
            )
        ):
            _fail(
                f"Cohort {cohort_name!r} has invalid layers: {layers!r}"
            )
        if cohort_name in cohort_layers:
            _fail(f"Duplicate cohort name: {cohort_name!r}")
        cohort_layers[cohort_name] = tuple(layers)

    declared_layers = tuple(
        layer
        for layers in cohort_layers.values()
        for layer in layers
    )
    if (
        len(declared_layers) != layer_count
        or len(set(declared_layers)) != layer_count
    ):
        _fail(
            "runtime_ready layer descriptors do not match layer_count: "
            f"{declared_layers!r}"
        )

    registered_events = _events_named(
        events,
        "hot_cache_registered",
    )
    if len(registered_events) != layer_count:
        _fail(
            "Expected one hot_cache_registered event per layer, got "
            f"{len(registered_events)} for {layer_count} layers"
        )
    registered_ptrs: dict[str, tuple[int, ...]] = {}
    registered_shapes: dict[str, object] = {}
    all_registered_ptrs: set[int] = set()
    for event in registered_events:
        layer = _require_string(
            event.get("layer"),
            "hot_cache_registered.layer",
        )
        if layer not in declared_layers:
            _fail(f"Registered unexpected Hot Cache layer: {layer!r}")
        if layer in registered_ptrs:
            _fail(f"Hot Cache layer registered twice: {layer!r}")
        pointers = _require_int_list(
            event.get("hot_cache_ptrs"),
            f"{layer}.hot_cache_ptrs",
        )
        if any(pointer <= 0 for pointer in pointers):
            _fail(f"Layer {layer!r} has an invalid Hot Cache address.")
        reused = all_registered_ptrs.intersection(pointers)
        if reused:
            _fail(
                f"Hot Cache addresses are shared across layers: {reused}"
            )
        registered_ptrs[layer] = pointers
        registered_shapes[layer] = event.get("hot_cache_shapes")
        all_registered_ptrs.update(pointers)

    completion_tokens = _read_completion_tokens(response_json)
    lookup_events = _events_named(events, "lookup_update_done")
    expected_lookup_events = cohort_count * completion_tokens
    if len(lookup_events) != expected_lookup_events:
        _fail(
            "lookup_update_done count mismatch: expected "
            f"{expected_lookup_events}, got {len(lookup_events)}"
        )
    lookup_counts = Counter(
        _require_string(
            event.get("cohort"),
            "lookup_update_done.cohort",
        )
        for event in lookup_events
    )
    expected_lookup_counts = {
        cohort: completion_tokens
        for cohort in cohort_layers
    }
    if dict(lookup_counts) != expected_lookup_counts:
        _fail(
            "lookup_update_done cohort counts mismatch: expected "
            f"{expected_lookup_counts}, got {dict(lookup_counts)}"
        )
    for event in lookup_events:
        cohort = _require_string(
            event.get("cohort"),
            "lookup_update_done.cohort",
        )
        if cohort not in cohort_layers:
            _fail(
                "lookup_update_done names an undeclared cohort: "
                f"{cohort!r}"
            )
        if event.get("role") != "target":
            _fail(
                "lookup_update_done.role must be 'target', got "
                f"{event.get('role')!r}"
            )
        req_num = _require_positive_int(
            event.get("req_num"),
            "lookup_update_done.req_num",
        )
        expected_matrix_shape = (req_num, index_topk)
        expected_vector_shape = (req_num,)
        matrix_shapes = {
            field: _require_int_list(
                event.get(field),
                f"lookup_update_done.{field}",
            )
            for field in (
                "query_index_shape",
                "lookup_mask_shape",
                "slot_out_shape",
                "miss_out_shape",
            )
        }
        for field, shape in matrix_shapes.items():
            if shape != expected_matrix_shape:
                _fail(
                    f"lookup_update_done.{field} must be "
                    f"{expected_matrix_shape}, got {shape!r}"
                )
        req_pool_entries_shape = _require_int_list(
            event.get("req_pool_entries_shape"),
            "lookup_update_done.req_pool_entries_shape",
        )
        if req_pool_entries_shape != expected_vector_shape:
            _fail(
                "lookup_update_done.req_pool_entries_shape must be "
                f"{expected_vector_shape}, got "
                f"{req_pool_entries_shape!r}"
            )

    hot_sfa_events = _events_named(
        events,
        "hot_cache_sfa_done",
    )
    expected_hot_sfa_events = layer_count * completion_tokens
    if len(hot_sfa_events) != expected_hot_sfa_events:
        _fail(
            "hot_cache_sfa_done count mismatch: expected "
            f"{expected_hot_sfa_events}, got {len(hot_sfa_events)}"
        )
    hot_sfa_counts: Counter[str] = Counter()
    for event in hot_sfa_events:
        layer = _require_string(
            event.get("layer"),
            "hot_cache_sfa_done.layer",
        )
        if layer not in registered_ptrs:
            _fail(f"Hot Cache SFA used an unregistered layer: {layer!r}")
        hot_sfa_counts[layer] += 1
        pointers = _require_int_list(
            event.get("hot_cache_ptrs"),
            f"{layer}.hot_cache_sfa_done.hot_cache_ptrs",
        )
        if pointers != registered_ptrs[layer]:
            _fail(
                f"Layer {layer!r} SFA did not use its registered Hot Cache: "
                f"expected {registered_ptrs[layer]}, got {pointers}"
            )
        if event.get("hot_cache_shapes") != registered_shapes[layer]:
            _fail(
                f"Layer {layer!r} Hot Cache shape changed before SFA."
            )
        sparse_shape = _require_int_list(
            event.get("sparse_indices_shape"),
            f"{layer}.sparse_indices_shape",
        )
        if (
            len(sparse_shape) != 3
            or sparse_shape[1] != 1
            or sparse_shape[2] != index_topk
        ):
            _fail(
                f"Layer {layer!r} has an invalid SFA sparse index shape: "
                f"{sparse_shape!r}"
            )
        block_table_shape = _require_int_list(
            event.get("hot_block_table_shape"),
            f"{layer}.hot_block_table_shape",
        )
        if len(block_table_shape) != 2:
            _fail(
                f"Layer {layer!r} has an invalid Hot Block Table shape: "
                f"{block_table_shape!r}"
            )
        _require_positive_int(
            event.get("hot_block_table_ptr"),
            f"{layer}.hot_block_table_ptr",
        )

    expected_hot_sfa_counts = {
        layer: completion_tokens
        for layer in declared_layers
    }
    if dict(hot_sfa_counts) != expected_hot_sfa_counts:
        _fail(
            "hot_cache_sfa_done layer counts mismatch: expected "
            f"{expected_hot_sfa_counts}, got {dict(hot_sfa_counts)}"
        )

    if not _profile_contains_custom_op(profile_dir):
        _fail(
            "Decode profile does not contain DsaSparseLookupUpdate, "
            "dsa_sparse_lookup_update, or aclnnDsaSparseLookupUpdate."
        )

    return {
        "completion_tokens": completion_tokens,
        "cohort_count": cohort_count,
        "layer_count": layer_count,
        "lookup_update_done": len(lookup_events),
        "hot_cache_sfa_done": len(hot_sfa_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the DSA Sparse custom-op and per-layer Hot Cache "
            "execution path."
        )
    )
    parser.add_argument(
        "--decode-log",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--response-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    summary = validate(
        decode_log=args.decode_log,
        response_json=args.response_json,
        profile_dir=args.profile_dir,
    )
    print(
        "PASS: DSA Sparse custom operator and Hot Cache path verified: "
        + json.dumps(summary, sort_keys=True)
    )


if __name__ == "__main__":
    main()
