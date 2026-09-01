# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import csv
import json
from pathlib import Path

from examples.dsa_offload_mtp_kvgathersim_graph_validate import (
    GRAPH_MARKER,
    validate,
)


def test_graph_profile_validator_accepts_required_runtime_evidence(
    tmp_path: Path,
) -> None:
    decode_log = tmp_path / "decode.log"
    response_json = tmp_path / "response.json"
    trace_dir = tmp_path / "profile/rank0_ascend_pt"
    trace_dir.mkdir(parents=True)
    decode_log.write_text(
        GRAPH_MARKER + "\nReplaying aclgraph\n",
        encoding="utf-8",
    )
    response_json.write_text(
        json.dumps(
            {
                "choices": [{"text": "x"}] * 2,
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    with (trace_dir / "kernel_details.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["Name"])
        writer.writerow(["dsa_offload_lookup_update"])
        writer.writerow(["aclnnAsuKvGather"])
        writer.writerow(["SparseFlashAttention"])

    summary = validate(
        decode_log,
        response_json,
        trace_dir.parent,
        batch_size=2,
        prompt_tokens=4,
        output_tokens=3,
    )

    assert summary["decode_graph"] == "FULL_DECODE_ONLY"
    assert summary["decode_graph_replayed"] is True
    assert summary["profile_evidence"] == {
        "lookup_update": True,
        "asu_kv_gather": True,
        "sparse_flash_attention": True,
    }
