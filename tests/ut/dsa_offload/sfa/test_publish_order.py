# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch

from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout
from vllm_ascend.dsa_offload.lookup import DSAOffloadBatch, IndexCacheCohort
from vllm_ascend.dsa_offload.pd import PrefillPublishState
from vllm_ascend.dsa_offload.sfa import publish_prefill_layer


class OrderedIO:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def put_blocks(self, **kwargs) -> None:
        self.events.append("put")


def test_final_prefill_order_is_indexer_put_capture_then_sfa() -> None:
    events = ["indexer"]
    io = OrderedIO(events)
    cohort = IndexCacheCohort("layer", "layer", ("layer",), (2,))
    publish = PrefillPublishState(
        request_ids=("request",),
        scheduled_token_counts=(1,),
        stored_token_counts=(4,),
        publish_requests=(True,),
        committed_block_keys={"request": [101]},
        io_backend=io,
        tp_rank=0,
    )
    batch = DSAOffloadBatch(
        layout=HotCacheLayout(4, 1, 1),
        hot_cache=None,
        io_backend=io,
        cohorts=(cohort,),
        lookup_states={},
        request_ids=("request",),
        request_rows=torch.tensor([-1], dtype=torch.int32),
        request_rows_cpu=(-1,),
        decode_request_indices=(),
        query_ranges=((0, 1),),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_positions=torch.tensor([3]),
        query_positions_cpu=(3,),
        is_mtp=False,
        committed_block_keys={"request": [101]},
        candidate_block_keys={},
        prefill_state=publish,
    )

    publish_prefill_layer(
        layer_name="layer",
        semantic_topk=torch.tensor([[3, 2, 1, 0]], dtype=torch.int32),
        main_cache=(torch.empty((1, 4, 1)),),
        block_table=torch.tensor([[7]], dtype=torch.int32),
        batch=batch,
    )
    events.append("sfa")

    assert events == ["indexer", "put", "sfa"]
    assert publish.layer_topk["layer"]["request"] == [3, 2, 1, 0]
