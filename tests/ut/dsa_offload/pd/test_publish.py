# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.dsa_offload.constants import QUERY_WIDTH
from vllm_ascend.dsa_offload.io import make_storage_id
from vllm_ascend.dsa_offload.pd import PrefillPublishState


def test_final_prefill_puts_full_blocks_and_captures_last_topk_and_tail(spy_io) -> None:
    semantic_topk = torch.arange(4 * QUERY_WIDTH, dtype=torch.int32).reshape(4, QUERY_WIDTH)
    state = PrefillPublishState(
        request_ids=("final", "middle"),
        scheduled_token_counts=(2, 2),
        stored_token_counts=(6, 5),
        publish_requests=(True, False),
        committed_block_hashes={"final": [b"full-0"], "middle": [b"unused"]},
        io_backend=spy_io,
        tp_rank=1,
    )

    state.publish_layer(
        layer_name="layer",
        layer_id=7,
        semantic_topk=semantic_topk,
        main_cache=(torch.empty((4, 4, 1)),),
        block_table=torch.tensor([[2, 3], [8, 9]], dtype=torch.int32),
    )

    assert len(spy_io.put_calls) == 1
    assert spy_io.put_calls[0]["layer_id"] == 7
    assert spy_io.put_calls[0]["source_block_ids"].tolist() == [2]
    assert spy_io.put_calls[0]["storage_ids"].tolist() == [make_storage_id(b"full-0", 7)]
    assert state.layer_topk["layer"]["final"] == semantic_topk[1].tolist()
    assert state.partial_tail_blocks["layer"] == {"final": 3}

    metadata = state.worker_metadata()
    assert metadata is not None
    assert metadata.request_layer_topk_by_rank["final"][1]["layer"] == semantic_topk[1].tolist()
    assert metadata.request_partial_tail_blocks_by_rank["final"][1]["layer"] == 3

    local = state.local_handoffs(block_size=4)
    assert len(local) == 1
    assert local[0].request_id == "final"
    assert local[0].stored_token_count == 6
    assert local[0].layer_topk["layer"] == semantic_topk[1].tolist()
    assert local[0].partial_tail_blocks == {"layer": 3}


def test_intermediate_prefill_does_not_publish(spy_io) -> None:
    state = PrefillPublishState(
        request_ids=("request",),
        scheduled_token_counts=(2,),
        stored_token_counts=(2,),
        publish_requests=(False,),
        committed_block_hashes={"request": []},
        io_backend=spy_io,
        tp_rank=0,
    )
    state.publish_layer(
        layer_name="layer",
        layer_id=0,
        semantic_topk=torch.zeros((2, QUERY_WIDTH), dtype=torch.int32),
        main_cache=(torch.empty((1, 4, 1)),),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )

    assert spy_io.put_calls == []
    assert state.worker_metadata() is None


def test_short_final_prefill_captures_handoff_without_empty_put(spy_io) -> None:
    state = PrefillPublishState(
        request_ids=("request",),
        scheduled_token_counts=(2,),
        stored_token_counts=(2,),
        publish_requests=(True,),
        committed_block_hashes={"request": []},
        io_backend=spy_io,
        tp_rank=0,
    )
    semantic_topk = torch.zeros((2, QUERY_WIDTH), dtype=torch.int32)
    state.publish_layer(
        layer_name="layer",
        layer_id=0,
        semantic_topk=semantic_topk,
        main_cache=(torch.empty((1, 4, 1)),),
        block_table=torch.tensor([[7]], dtype=torch.int32),
    )

    assert spy_io.put_calls == []
    assert state.layer_topk["layer"]["request"] == semantic_topk[1].tolist()
    assert state.partial_tail_blocks["layer"] == {"request": 7}


def test_full_block_publish_rejects_missing_hashes(spy_io) -> None:
    state = PrefillPublishState(
        request_ids=("request",),
        scheduled_token_counts=(4,),
        stored_token_counts=(4,),
        publish_requests=(True,),
        committed_block_hashes={"request": []},
        io_backend=spy_io,
        tp_rank=0,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Prefill publish for request request requires 1 block hashes",
    ):
        state.publish_layer(
            layer_name="layer",
            layer_id=0,
            semantic_topk=torch.zeros(
                (4, QUERY_WIDTH),
                dtype=torch.int32,
            ),
            main_cache=(torch.empty((1, 4, 1)),),
            block_table=torch.tensor([[7]], dtype=torch.int32),
        )
