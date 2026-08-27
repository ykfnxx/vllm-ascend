# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import patch

import torch

from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout, HotCacheState
from vllm_ascend.dsa_offload.lookup import (
    IndexCacheCohort,
    LookupPlan,
    build_dsa_offload_batch,
)
from vllm_ascend.dsa_offload.sfa import (
    prepare_main_slot_mapping,
    resolve_sfa_inputs,
)


def make_mixed_batch(spy_io, *, is_mtp: bool = False):
    layout = HotCacheLayout(4, 2, 3)
    cache = torch.empty((layout.hot_blocks, 4, 1))
    hot_cache = HotCacheState(layout, {"leader": (cache,), "follower": (cache,)})
    row = hot_cache.admit("decode")
    cohort = IndexCacheCohort(
        "leader",
        "leader",
        ("leader", "follower"),
        (3, 4),
    )
    batch = build_dsa_offload_batch(
        layout=layout,
        hot_cache=hot_cache,
        io_backend=spy_io,
        cohorts=(cohort,),
        lookup_states={},
        request_ids=("prefill", "decode"),
        query_counts=(2, 2 if is_mtp else 1),
        query_positions=torch.tensor(
            [0, 1, 5, 6] if is_mtp else [0, 1, 5],
            dtype=torch.int64,
        ),
        is_mtp=is_mtp,
        committed_block_hashes={"prefill": [], "decode": [b"block"]},
        candidate_block_hashes={},
    )
    return batch, row


def test_feature_off_returns_original_inputs() -> None:
    slots = torch.tensor([1, 2], dtype=torch.int64)
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    table = torch.tensor([[3, 4]], dtype=torch.int32)

    assert prepare_main_slot_mapping(batch=None, default_slot_mapping=slots) is slots
    resolved = resolve_sfa_inputs(
        layer_name="layer",
        semantic_topk=topk,
        default_block_table=table,
        batch=None,
    )
    assert resolved == (topk, table)


def test_mixed_batch_keeps_prefill_mapping_and_redirects_decode_tail(spy_io) -> None:
    batch, row = make_mixed_batch(spy_io)
    default = torch.tensor([10, 11, 12], dtype=torch.int64)

    mapped = prepare_main_slot_mapping(
        batch=batch,
        default_slot_mapping=default,
    )

    assert mapped[:2].tolist() == [10, 11]
    assert mapped[2].item() == batch.layout.global_slot(row, batch.layout.tail_base + 1)
    assert default.tolist() == [10, 11, 12]


def test_mtp_verification_writes_staging_without_touching_prefill(spy_io) -> None:
    batch, row = make_mixed_batch(spy_io, is_mtp=True)
    mapped = prepare_main_slot_mapping(
        batch=batch,
        default_slot_mapping=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
    )

    assert mapped.tolist() == [
        10,
        11,
        batch.layout.global_slot(row, batch.layout.staging_base),
        batch.layout.global_slot(row, batch.layout.staging_base + 1),
    ]


def test_leader_looks_up_once_and_follower_performs_own_get(spy_io) -> None:
    batch, _ = make_mixed_batch(spy_io)
    mapped = torch.tensor([[8, 9]], dtype=torch.int32)
    table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    hot_table = torch.tensor([[1, 2], [9, 10]], dtype=torch.int32)
    empty = torch.empty(0, dtype=torch.int64)
    plan = LookupPlan(
        mapped_indices=mapped,
        miss_positions=empty,
        miss_logical_blocks=empty,
        miss_block_offsets=empty,
        miss_destination_slots=empty,
        miss_batch_indices=empty.to(torch.int32),
        query_request_rows=empty.to(torch.int32),
        hot_block_table=hot_table,
        tail_mask=torch.empty(0, dtype=torch.bool),
        fallback_mask=torch.empty(0, dtype=torch.bool),
        staging_mask=torch.empty(0, dtype=torch.bool),
    )
    events = ["indexer"]

    def make_plan(**kwargs):
        events.append("lookup")
        return plan

    def load_misses(plan, layer_id, batch):
        events.append(f"get:{layer_id}")

    with (
        patch("vllm_ascend.dsa_offload.lookup.make_lookup_plan", side_effect=make_plan) as lookup,
        patch("vllm_ascend.dsa_offload.lookup.load_plan_misses", side_effect=load_misses),
    ):
        leader_topk, leader_table = resolve_sfa_inputs(
            layer_name="leader",
            semantic_topk=mapped,
            default_block_table=table,
            batch=batch,
        )
        assert leader_topk is mapped
        assert torch.equal(leader_table, hot_table)
        events.append("sfa:leader")
        follower_default = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
        follower_topk, follower_table = resolve_sfa_inputs(
            layer_name="follower",
            semantic_topk=mapped,
            default_block_table=follower_default,
            batch=batch,
        )
        assert follower_topk is mapped
        assert follower_table.tolist() == [[5, 6], [9, 10]]
        events.append("sfa:follower")

    lookup.assert_called_once()
    assert events == [
        "indexer",
        "lookup",
        "get:3",
        "sfa:leader",
        "get:4",
        "sfa:follower",
    ]
