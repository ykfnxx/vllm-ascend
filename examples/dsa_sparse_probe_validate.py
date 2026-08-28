# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROBE_LOG_PREFIX = "DSA_SPARSE_PROBE "
SINGLE_QUERY_OP_NAMES = (
    "dsasparselookupupdate",
    "dsa_sparse_lookup_update",
    "aclnndsasparselookupupdate",
)
BATCH_OP_NAMES = (
    "dsasparselookupupdatebatch",
    "dsa_sparse_lookup_update_batch",
    "aclnndsasparselookupupdatebatch",
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


def _profile_operator_modes(
    profile_dir: Path,
) -> set[str]:
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
    modes: set[str] = set()
    for profile_file in profile_files:
        with profile_file.open(
            encoding="utf-8",
            errors="replace",
        ) as csv_file:
            for line in csv_file:
                lowered = line.lower()
                if any(op_name in lowered for op_name in BATCH_OP_NAMES):
                    modes.add("mtp_batch")
                elif any(
                    op_name in lowered
                    for op_name in SINGLE_QUERY_OP_NAMES
                ):
                    modes.add("single_query")
    return modes


def _require_step_id(event: dict[str, Any], field: str) -> int:
    value = event.get("target_step_id")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer, got {value!r}")
    return value


def _require_q_i(
    event: dict[str, Any],
    *,
    field: str,
    req_num: int,
) -> tuple[int, ...]:
    q_i = _require_int_list(event.get("q_i"), field)
    if len(q_i) != req_num or any(query_len <= 0 for query_len in q_i):
        _fail(
            f"{field} must contain {req_num} positive query lengths, "
            f"got {q_i!r}"
        )
    return q_i


def _validate_mtp_batch(
    *,
    events: list[dict[str, Any]],
    completion_tokens: int,
    cohort_layers: dict[str, tuple[str, ...]],
    declared_layers: tuple[str, ...],
    registered_ptrs: dict[str, tuple[int, ...]],
    registered_shapes: dict[str, object],
    index_topk: int,
    profile_dir: Path,
) -> dict[str, int]:
    lookup_events = _events_named(events, "lookup_update_done")
    if not lookup_events:
        _fail("No lookup_update_done events found for MTP target verify.")

    lookup_by_step: dict[int, dict[str, dict[str, Any]]] = {}
    step_query_lens: dict[int, tuple[int, ...]] = {}
    step_token_counts: dict[int, int] = {}
    for event in lookup_events:
        if event.get("operator") != "dsa_sparse_lookup_update_batch":
            _fail(
                "MTP lookup_update_done.operator must be "
                "'dsa_sparse_lookup_update_batch', got "
                f"{event.get('operator')!r}"
            )
        if event.get("role") != "target":
            _fail(
                "lookup_update_done.role must be 'target', got "
                f"{event.get('role')!r}"
            )
        step_id = _require_step_id(event, "lookup_update_done.target_step_id")
        cohort = _require_string(
            event.get("cohort"),
            "lookup_update_done.cohort",
        )
        if cohort not in cohort_layers:
            _fail(f"lookup_update_done names an undeclared cohort: {cohort!r}")
        step_events = lookup_by_step.setdefault(step_id, {})
        if cohort in step_events:
            _fail(
                f"Cohort {cohort!r} executed batch lookup more than once "
                f"in target step {step_id}."
            )
        step_events[cohort] = event

        req_num = _require_positive_int(
            event.get("req_num"),
            "lookup_update_done.req_num",
        )
        q_i = _require_q_i(
            event,
            field="lookup_update_done.q_i",
            req_num=req_num,
        )
        token_count = sum(q_i)
        if step_id in step_query_lens and step_query_lens[step_id] != q_i:
            _fail(
                f"MTP cohorts disagree on q_i in target step {step_id}: "
                f"{step_query_lens[step_id]!r} versus {q_i!r}"
            )
        step_query_lens[step_id] = q_i
        step_token_counts[step_id] = token_count

        expected_matrix_shape = (token_count, index_topk)
        for field in (
            "query_index_shape",
            "lookup_mask_shape",
            "slot_out_shape",
            "miss_out_shape",
        ):
            shape = _require_int_list(
                event.get(field),
                f"lookup_update_done.{field}",
            )
            if shape != expected_matrix_shape:
                _fail(
                    f"lookup_update_done.{field} must be "
                    f"{expected_matrix_shape}, got {shape!r}"
                )
        req_pool_shape = _require_int_list(
            event.get("req_pool_entries_shape"),
            "lookup_update_done.req_pool_entries_shape",
        )
        if req_pool_shape != (req_num,):
            _fail(
                "lookup_update_done.req_pool_entries_shape must be "
                f"{(req_num,)}, got {req_pool_shape!r}"
            )
        for counter_name in (
            "history_miss_count",
            "fallback_overflow_count",
        ):
            counter = event.get(counter_name)
            if (
                isinstance(counter, bool)
                or not isinstance(counter, int)
                or counter < 0
                or counter > token_count * index_topk
            ):
                _fail(
                    f"lookup_update_done.{counter_name} is invalid: "
                    f"{counter!r}"
                )

    expected_cohorts = set(cohort_layers)
    layer_to_cohort = {
        layer: cohort
        for cohort, layers in cohort_layers.items()
        for layer in layers
    }
    for step_id, step_events in lookup_by_step.items():
        if set(step_events) != expected_cohorts:
            _fail(
                f"Target step {step_id} did not execute exactly one batch "
                f"lookup per cohort: got {set(step_events)!r}, expected "
                f"{expected_cohorts!r}"
            )
    step_ids = set(lookup_by_step)

    hot_sfa_events = _events_named(events, "hot_cache_sfa_done")
    hot_sfa_pairs: set[tuple[int, str]] = set()
    for event in hot_sfa_events:
        if event.get("execution_mode") != "mtp_batch":
            _fail(
                "MTP hot_cache_sfa_done.execution_mode must be "
                f"'mtp_batch', got {event.get('execution_mode')!r}"
            )
        step_id = _require_step_id(event, "hot_cache_sfa_done.target_step_id")
        if step_id not in step_ids:
            _fail(f"Hot Cache SFA names unknown target step {step_id}.")
        layer = _require_string(event.get("layer"), "hot_cache_sfa_done.layer")
        if layer not in registered_ptrs:
            _fail(f"Hot Cache SFA used an unregistered layer: {layer!r}")
        pair = (step_id, layer)
        if pair in hot_sfa_pairs:
            _fail(f"Layer {layer!r} ran Hot Cache SFA twice in step {step_id}.")
        hot_sfa_pairs.add(pair)
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
            _fail(f"Layer {layer!r} Hot Cache shape changed before SFA.")
        sparse_shape = _require_int_list(
            event.get("sparse_indices_shape"),
            f"{layer}.sparse_indices_shape",
        )
        expected_sparse_shape = (step_token_counts[step_id], 1, index_topk)
        if sparse_shape != expected_sparse_shape:
            _fail(
                f"Layer {layer!r} sparse indices must be "
                f"{expected_sparse_shape}, got {sparse_shape!r}"
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
    expected_layer_step_pairs = {
        (step_id, layer)
        for step_id in step_ids
        for layer in declared_layers
    }
    if hot_sfa_pairs != expected_layer_step_pairs:
        _fail(
            "hot_cache_sfa_done did not cover every physical layer exactly "
            f"once per target step: got {hot_sfa_pairs!r}, expected "
            f"{expected_layer_step_pairs!r}"
        )

    history_events = _events_named(events, "history_load_mock")
    history_pairs: set[tuple[int, str]] = set()
    for event in history_events:
        step_id = _require_step_id(event, "history_load_mock.target_step_id")
        layer = _require_string(event.get("layer"), "history_load_mock.layer")
        pair = (step_id, layer)
        if step_id not in step_ids or layer not in registered_ptrs:
            _fail(f"history_load_mock names an unknown step/layer: {pair!r}")
        if pair in history_pairs:
            _fail(f"history_load_mock duplicated step/layer {pair!r}")
        history_pairs.add(pair)
        if event.get("cohort") != layer_to_cohort[layer]:
            _fail(
                f"history_load_mock used the wrong cohort for layer "
                f"{layer!r}: {event.get('cohort')!r}"
            )
        q_i = _require_q_i(
            event,
            field="history_load_mock.q_i",
            req_num=len(step_query_lens[step_id]),
        )
        if q_i != step_query_lens[step_id]:
            _fail(f"history_load_mock q_i disagrees in target step {step_id}.")
    if history_pairs != expected_layer_step_pairs:
        _fail(
            "history_load_mock did not cover every physical layer exactly "
            "once per target step."
        )

    store_events = _events_named(events, "accepted_store_mock")
    store_pairs: set[tuple[int, str]] = set()
    step_accepted: dict[int, tuple[int, ...]] = {}
    for event in store_events:
        step_id = _require_step_id(event, "accepted_store_mock.target_step_id")
        layer = _require_string(event.get("layer"), "accepted_store_mock.layer")
        pair = (step_id, layer)
        if step_id not in step_ids or layer not in registered_ptrs:
            _fail(f"accepted_store_mock names an unknown step/layer: {pair!r}")
        if pair in store_pairs:
            _fail(f"accepted_store_mock duplicated step/layer {pair!r}")
        store_pairs.add(pair)
        if event.get("cohort") != layer_to_cohort[layer]:
            _fail(
                f"accepted_store_mock used the wrong cohort for layer "
                f"{layer!r}: {event.get('cohort')!r}"
            )
        q_i = _require_q_i(
            event,
            field="accepted_store_mock.q_i",
            req_num=len(step_query_lens[step_id]),
        )
        if q_i != step_query_lens[step_id]:
            _fail(f"accepted_store_mock q_i disagrees in target step {step_id}.")
        accepted = _require_int_list(
            event.get("accepted_input_kv_count"),
            "accepted_store_mock.accepted_input_kv_count",
        )
        if len(accepted) != len(q_i) or any(
            count < 0 or count > query_len
            for count, query_len in zip(accepted, q_i)
        ):
            _fail(
                "accepted_store_mock.accepted_input_kv_count must be a "
                f"per-request prefix bounded by q_i, got {accepted!r}"
            )
        if event.get("committed_kv_count") != sum(accepted):
            _fail("accepted_store_mock.committed_kv_count is inconsistent.")
        req_pool_entries = _require_int_list(
            event.get("req_pool_entries"),
            "accepted_store_mock.req_pool_entries",
        )
        if (
            len(req_pool_entries) != len(q_i)
            or len(set(req_pool_entries)) != len(req_pool_entries)
            or any(entry < 0 for entry in req_pool_entries)
        ):
            _fail(
                "accepted_store_mock.req_pool_entries must contain one "
                "unique non-negative row per request."
            )
        committed_ranges = event.get("committed_position_ranges")
        staging_ranges = event.get("staging_source_slot_ranges")
        if (
            not isinstance(committed_ranges, list)
            or len(committed_ranges) != len(q_i)
            or not isinstance(staging_ranges, list)
            or len(staging_ranges) != len(q_i)
        ):
            _fail("accepted_store_mock prefix range metadata is invalid.")
        for count, position_range, staging_range in zip(
            accepted,
            committed_ranges,
            staging_ranges,
        ):
            if count == 0:
                if position_range is not None:
                    _fail("A zero-length commit must have no position range.")
            elif (
                not isinstance(position_range, list)
                or len(position_range) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in position_range
                )
                or position_range[1] - position_range[0] != count
            ):
                _fail("accepted_store_mock committed position range is invalid.")
            if (
                not isinstance(staging_range, list)
                or len(staging_range) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in staging_range
                )
                or staging_range[1] - staging_range[0] != count
            ):
                _fail("accepted_store_mock staging source range is invalid.")
        if step_id in step_accepted and step_accepted[step_id] != accepted:
            _fail(
                f"Physical layers disagree on accepted prefix in step {step_id}."
            )
        step_accepted[step_id] = accepted
    if store_pairs != expected_layer_step_pairs:
        _fail(
            "accepted_store_mock did not cover every physical layer exactly "
            "once per target step."
        )

    if "mtp_batch" not in _profile_operator_modes(profile_dir):
        _fail("Decode profile does not contain dsa_sparse_lookup_update_batch.")

    return {
        "completion_tokens": completion_tokens,
        "cohort_count": len(cohort_layers),
        "layer_count": len(declared_layers),
        "target_steps": len(step_ids),
        "lookup_update_done": len(lookup_events),
        "history_load_mock": len(history_events),
        "hot_cache_sfa_done": len(hot_sfa_events),
        "accepted_store_mock": len(store_events),
    }


def validate(
    *,
    decode_log: Path,
    response_json: Path,
    profile_dir: Path,
) -> dict[str, int]:
    events = _read_probe_events(decode_log)
    runtime_events = _events_named(events, "coordinators_ready")
    if len(runtime_events) != 1:
        _fail(
            "Expected exactly one coordinators_ready event, got "
            f"{len(runtime_events)}"
        )
    runtime = runtime_events[0]
    cohort_count = _require_positive_int(
        runtime.get("cohort_count"),
        "coordinators_ready.cohort_count",
    )
    layer_count = _require_positive_int(
        runtime.get("layer_count"),
        "coordinators_ready.layer_count",
    )
    index_topk = _require_positive_int(
        runtime.get("index_topk"),
        "coordinators_ready.index_topk",
    )

    cohorts = runtime.get("cohorts")
    if not isinstance(cohorts, list) or len(cohorts) != cohort_count:
        _fail(
            "coordinators_ready.cohorts does not match cohort_count: "
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
            "coordinators_ready layer descriptors do not match layer_count: "
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
    if any(
        event.get("operator") == "dsa_sparse_lookup_update_batch"
        for event in lookup_events
    ):
        return _validate_mtp_batch(
            events=events,
            completion_tokens=completion_tokens,
            cohort_layers=cohort_layers,
            declared_layers=declared_layers,
            registered_ptrs=registered_ptrs,
            registered_shapes=registered_shapes,
            index_topk=index_topk,
            profile_dir=profile_dir,
        )

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

    if "single_query" not in _profile_operator_modes(profile_dir):
        _fail(
            "Decode profile does not contain dsa_sparse_lookup_update."
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
