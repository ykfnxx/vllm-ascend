# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend import dsa_sparse_probe
from vllm_ascend.attention.dsa_sparse_pd import (
    DSA_SPARSE_PD_HANDOFF_KEY,
    DSASparsePDHandoff,
    begin_dsa_sparse_producer_execution,
    build_dsa_sparse_resident_token_ids,
    get_dsa_sparse_pd_handoff,
)
from vllm_ascend.dsa_sparse_constants import DSA_SPARSE_QUERY_WIDTH


def _make_handoff() -> DSASparsePDHandoff:
    return DSASparsePDHandoff(
        remote_request_id="prefill-request",
        stored_token_count=4097,
        block_size=128,
        layer_topk_by_rank={
            0: {
                "model.layers.0.self_attn": list(
                    range(DSA_SPARSE_QUERY_WIDTH)
                ),
            },
        },
    )


def test_resident_selection_preserves_topk_order_and_excludes_dense_tail():
    resident = build_dsa_sparse_resident_token_ids(
        topk_token_ids=[255, 7, 3, 7, -1, 256, 128, 200],
        stored_token_count=259,
        block_size=128,
        resident_token_count=6,
    )

    assert resident == [255, 7, 3, 128, 200, 0]


def test_pd_handoff_round_trips_through_transfer_params():
    handoff = _make_handoff()
    transfer_params = {DSA_SPARSE_PD_HANDOFF_KEY: handoff.to_dict()}

    assert DSASparsePDHandoff.from_dict(handoff.to_dict()) == handoff
    assert get_dsa_sparse_pd_handoff(transfer_params) == handoff


def test_pd_handoff_rejects_non_integer_topk_ids():
    raw_handoff = _make_handoff().to_dict()
    raw_handoff["layer_topk_by_rank"]["0"][
        "model.layers.0.self_attn"
    ] = [True] * DSA_SPARSE_QUERY_WIDTH

    with pytest.raises(TypeError, match="token IDs"):
        DSASparsePDHandoff.from_dict(raw_handoff)


def test_producer_execution_captures_only_final_prefill_rows_and_detaches():
    metadata = SimpleNamespace(dsa_sparse_producer_context=None)
    layer_name = "model.layers.0.self_attn"
    execution = begin_dsa_sparse_producer_execution(
        request_ids=["request-a", "request-b"],
        scheduled_token_counts=[2, 3],
        publish_requests=[False, True],
        layer_metadata={layer_name: metadata},
    )
    topk = torch.arange(
        5 * DSA_SPARSE_QUERY_WIDTH,
        dtype=torch.int32,
    ).reshape(5, 1, DSA_SPARSE_QUERY_WIDTH)

    with execution as context:
        assert metadata.dsa_sparse_producer_context is context
        context.publish_layer(layer_name, topk)
        assert context.layer_topk(layer_name) == {
            "request-b": topk[4].reshape(-1).tolist()
        }

    assert metadata.dsa_sparse_producer_context is None


def test_producer_execution_rejects_incomplete_topk_rows():
    metadata = SimpleNamespace(dsa_sparse_producer_context=None)
    layer_name = "model.layers.0.self_attn"
    execution = begin_dsa_sparse_producer_execution(
        request_ids=["request-a"],
        scheduled_token_counts=[2],
        publish_requests=[True],
        layer_metadata={layer_name: metadata},
    )

    with execution as context:
        with pytest.raises(ValueError, match="do not cover"):
            context.publish_layer(
                layer_name,
                torch.zeros(
                    (1, 1, DSA_SPARSE_QUERY_WIDTH),
                    dtype=torch.int32,
                ),
            )


def test_pd_handoff_trace_logs_full_handoff_and_stable_hashes():
    handoff = _make_handoff()

    with (
        patch.object(
            dsa_sparse_probe,
            "pd_trace_is_enabled",
            return_value=True,
        ),
        patch.object(dsa_sparse_probe.logger, "info") as log_info,
    ):
        dsa_sparse_probe.emit_pd_handoff(
            "handoff_send",
            role="P",
            request_id="prefill-request",
            handoff=handoff,
            tp_size=1,
        )

    prefix, encoded_payload = log_info.call_args.args
    payload = json.loads(encoded_payload)
    assert prefix == dsa_sparse_probe.DSA_SPARSE_PD_LOG_PREFIX
    assert payload["event"] == "handoff_send"
    assert payload["handoff"] == handoff.to_dict()
    assert payload["handoff_sha256"]
    assert payload["layer_topk_sha256_by_rank"]["0"][
        "model.layers.0.self_attn"
    ]
