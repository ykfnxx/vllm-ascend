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
from vllm_ascend.attention.dsa_sparse_shm import (
    DSASparseSharedMemoryPayload,
    DSASparseSharedMemoryPlane,
    DSASparseSharedMemoryStore,
)
from vllm_ascend.dsa_sparse_constants import DSA_SPARSE_QUERY_WIDTH


def _make_handoff() -> DSASparsePDHandoff:
    layer_name = "model.layers.0.self_attn"
    payload = DSASparseSharedMemoryPayload(
        name="vllm_ascend_dsa_sparse_test",
        size=4,
        cache_kind="indexer",
        cache_layer_name="model.layers.0.self_attn.indexer.k_cache",
        cache_planes=(
            DSASparseSharedMemoryPlane(
                offset=0,
                nbytes=2,
                dtype="bfloat16",
                shape=(1, 1, 1),
            ),
        ),
        tail_planes=(
            DSASparseSharedMemoryPlane(
                offset=2,
                nbytes=2,
                dtype="bfloat16",
                shape=(1, 1, 1),
            ),
        ),
    )
    return DSASparsePDHandoff(
        remote_request_id="prefill-request",
        stored_token_count=4097,
        block_size=128,
        layer_topk_by_rank={
            0: {
                layer_name: list(range(DSA_SPARSE_QUERY_WIDTH)),
            },
        },
        shared_memory_payloads_by_rank={0: {layer_name: payload}},
    )


def test_resident_selection_preserves_topk_order_and_excludes_dense_tail():
    resident = build_dsa_sparse_resident_token_ids(
        topk_token_ids=[255, 7, 3, 7, -1, 256, 128, 200],
        stored_token_count=259,
        block_size=128,
        resident_token_count=6,
    )

    assert resident == [255, 7, 3, 128, 200, 0]


def test_mtp_resident_selection_excludes_the_partial_prompt_tail():
    resident = build_dsa_sparse_resident_token_ids(
        topk_token_ids=[258, 257, 255],
        stored_token_count=259,
        block_size=128,
        resident_token_count=4,
    )

    assert resident == [255, 0, 1, 2]


def test_pd_handoff_round_trips_through_transfer_params():
    handoff = _make_handoff()
    transfer_params = {DSA_SPARSE_PD_HANDOFF_KEY: handoff.to_dict()}

    assert DSASparsePDHandoff.from_dict(handoff.to_dict()) == handoff
    assert get_dsa_sparse_pd_handoff(transfer_params) == handoff


def test_pd_handoff_rejects_non_integer_topk_ids():
    raw_handoff = _make_handoff().to_dict()
    raw_handoff["layer_topk_by_rank"]["0"]["model.layers.0.self_attn"] = [True] * DSA_SPARSE_QUERY_WIDTH

    with pytest.raises(TypeError, match="token IDs"):
        DSASparsePDHandoff.from_dict(raw_handoff)


def test_producer_execution_captures_only_final_prefill_rows_and_detaches(
    tmp_path,
):
    metadata = SimpleNamespace(dsa_sparse_producer_context=None)
    layer_name = "model.layers.0.self_attn"
    shared_memory_store = DSASparseSharedMemoryStore(tmp_path)
    execution = begin_dsa_sparse_producer_execution(
        request_ids=["request-a", "request-b"],
        scheduled_token_counts=[2, 3],
        stored_token_counts=[2, 3],
        publish_requests=[False, True],
        layer_metadata={layer_name: metadata},
        shared_memory_store=shared_memory_store,
    )
    topk = torch.arange(
        5 * DSA_SPARSE_QUERY_WIDTH,
        dtype=torch.int32,
    ).reshape(5, 1, DSA_SPARSE_QUERY_WIDTH)

    with execution as context:
        assert metadata.dsa_sparse_producer_context is context
        main_cache = (torch.empty((1, 128, 1, 1)),)
        indexer_cache = (torch.empty((1, 128, 1, 1)),)
        block_table = torch.zeros((2, 1), dtype=torch.int32)
        context.publish_layer(
            layer_name,
            topk,
            main_cache,
            block_table,
            "model.layers.0.self_attn.indexer.k_cache",
            indexer_cache,
        )
        assert context.layer_topk(layer_name) == {"request-b": topk[4].reshape(-1).tolist()}
        payload = context.layer_shared_memory_payloads(layer_name)[
            "request-b"
        ]
        draft_layer_name = "model.layers.1.self_attn"
        context.publish_mtp_draft_layer(
            draft_layer_name,
            (
                torch.arange(4 * 128, dtype=torch.bfloat16).reshape(
                    4,
                    128,
                    1,
                    1,
                ),
            ),
            torch.tensor([[0], [2]], dtype=torch.int32),
        )
        draft_payload = context.layer_shared_memory_payloads(
            draft_layer_name
        )["request-b"]
        assert draft_payload.cache_kind == "mtp_draft"
        assert context.layer_topk(draft_layer_name) == {}

    assert metadata.dsa_sparse_producer_context is None
    assert (tmp_path / payload.name).exists()
    assert (tmp_path / draft_payload.name).exists()
    shared_memory_store.unlink(payload)
    shared_memory_store.unlink(draft_payload)


def test_producer_execution_rejects_incomplete_topk_rows():
    metadata = SimpleNamespace(dsa_sparse_producer_context=None)
    layer_name = "model.layers.0.self_attn"
    execution = begin_dsa_sparse_producer_execution(
        request_ids=["request-a"],
        scheduled_token_counts=[2],
        stored_token_counts=[2],
        publish_requests=[True],
        layer_metadata={layer_name: metadata},
    )

    with execution as context, pytest.raises(ValueError, match="do not cover"):
        context.publish_layer(
            layer_name,
            torch.zeros(
                (1, 1, DSA_SPARSE_QUERY_WIDTH),
                dtype=torch.int32,
            ),
            (torch.empty((1, 128, 1, 1)),),
            torch.zeros((1, 1), dtype=torch.int32),
        )


def test_producer_execution_defers_ownership_until_mtp_finishes(tmp_path):
    metadata = SimpleNamespace(dsa_sparse_producer_context=None)
    layer_name = "model.layers.0.self_attn"
    draft_layer_name = "model.layers.1.self_attn"
    store = DSASparseSharedMemoryStore(tmp_path)
    execution = begin_dsa_sparse_producer_execution(
        request_ids=["request-a"],
        scheduled_token_counts=[2],
        stored_token_counts=[2],
        publish_requests=[True],
        layer_metadata={layer_name: metadata},
        shared_memory_store=store,
        defer_completion=True,
    )

    with execution as context:
        context.publish_layer(
            layer_name,
            torch.zeros(
                (2, 1, DSA_SPARSE_QUERY_WIDTH),
                dtype=torch.int32,
            ),
            (torch.zeros((2, 2, 1, 1), dtype=torch.bfloat16),),
            torch.tensor([[1]], dtype=torch.int32),
            "model.layers.0.self_attn.indexer.k_cache",
            (torch.zeros((2, 2, 1, 1), dtype=torch.bfloat16),),
        )

    assert execution.is_pending
    assert metadata.dsa_sparse_producer_context is context
    context.publish_mtp_draft_layer(
        draft_layer_name,
        (torch.zeros((2, 2, 1, 1), dtype=torch.bfloat16),),
        torch.tensor([[0]], dtype=torch.int32),
    )
    payloads = [
        context.layer_shared_memory_payloads(layer_name)["request-a"],
        context.layer_shared_memory_payloads(draft_layer_name)["request-a"],
    ]

    execution.finish()

    assert not execution.is_pending
    assert metadata.dsa_sparse_producer_context is None
    assert all((tmp_path / payload.name).exists() for payload in payloads)
    for payload in payloads:
        store.unlink(payload)


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
    assert payload["layer_topk_sha256_by_rank"]["0"]["model.layers.0.self_attn"]
