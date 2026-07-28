# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.dsa_sparse_probe_validate import (
    PROBE_LOG_PREFIX,
    validate,
)


def _write_fixture(
    root: Path,
    *,
    second_layer_ptr: int = 201,
) -> tuple[Path, Path, Path]:
    decode_log = root / "decode.log"
    response_json = root / "response.json"
    profile_dir = root / "profile"
    profile_dir.mkdir()
    events = [
        {
            "event": "hot_cache_registered",
            "cohort": "cohort-0",
            "layer": "layer-0",
            "hot_cache_ptrs": [101, 102],
            "hot_cache_shapes": [[8, 128, 4], [8, 128, 2]],
        },
        {
            "event": "hot_cache_registered",
            "cohort": "cohort-1",
            "layer": "layer-1",
            "hot_cache_ptrs": [second_layer_ptr, 202],
            "hot_cache_shapes": [[8, 128, 4], [8, 128, 2]],
        },
        {
            "event": "runtime_ready",
            "cohort_count": 2,
            "layer_count": 2,
            "index_topk": 4,
            "cohorts": [
                {"name": "cohort-0", "layers": ["layer-0"]},
                {"name": "cohort-1", "layers": ["layer-1"]},
            ],
        },
        {
            "event": "lookup_update_done",
            "cohort": "cohort-0",
            "role": "target",
            "topk_shape": [1, 4],
        },
        {
            "event": "lookup_update_done",
            "cohort": "cohort-1",
            "role": "target",
            "topk_shape": [1, 4],
        },
        {
            "event": "hot_cache_sfa_done",
            "layer": "layer-0",
            "hot_cache_ptrs": [101, 102],
            "hot_cache_shapes": [[8, 128, 4], [8, 128, 2]],
            "sparse_indices_shape": [1, 1, 4],
            "hot_block_table_ptr": 301,
            "hot_block_table_shape": [1, 8],
        },
        {
            "event": "hot_cache_sfa_done",
            "layer": "layer-1",
            "hot_cache_ptrs": [second_layer_ptr, 202],
            "hot_cache_shapes": [[8, 128, 4], [8, 128, 2]],
            "sparse_indices_shape": [1, 1, 4],
            "hot_block_table_ptr": 301,
            "hot_block_table_shape": [1, 8],
        },
    ]
    decode_log.write_text(
        "".join(
            f"INFO {PROBE_LOG_PREFIX}{json.dumps(event)}\n"
            for event in events
        ),
        encoding="utf-8",
    )
    response_json.write_text(
        json.dumps({"usage": {"completion_tokens": 1}}),
        encoding="utf-8",
    )
    (profile_dir / "operator_details.csv").write_text(
        "Name,Duration\nDsaSparseLookupUpdate,1\n",
        encoding="utf-8",
    )
    return decode_log, response_json, profile_dir


def test_validate_accepts_custom_op_and_per_layer_hot_cache_path(
    tmp_path: Path,
):
    decode_log, response_json, profile_dir = _write_fixture(
        tmp_path,
    )

    summary = validate(
        decode_log=decode_log,
        response_json=response_json,
        profile_dir=profile_dir,
    )

    assert summary == {
        "completion_tokens": 1,
        "cohort_count": 2,
        "layer_count": 2,
        "lookup_update_done": 2,
        "hot_cache_sfa_done": 2,
    }


def test_validate_rejects_hot_cache_address_reuse(
    tmp_path: Path,
):
    decode_log, response_json, profile_dir = _write_fixture(
        tmp_path,
        second_layer_ptr=101,
    )

    with pytest.raises(
        RuntimeError,
        match="Hot Cache addresses are shared across layers",
    ):
        validate(
            decode_log=decode_log,
            response_json=response_json,
            profile_dir=profile_dir,
        )
