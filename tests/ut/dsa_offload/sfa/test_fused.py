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
    SFAAddressingWorkspace,
    resolve_sfa_inputs,
)


def make_fused_batch(spy_io, *, enable_cohort_kvgather: bool, with_host_planes: bool = True):
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
    host_key = torch.zeros((1, 4, 8))
    host_rope = torch.zeros((1, 4, 2))
    host_table = torch.zeros((4, 2), dtype=torch.int32)
    spy_io.gather_history_misses = lambda **_: True
    if with_host_planes:
        spy_io.host_cache_planes = lambda *, layer_id: (host_key, host_rope)
        spy_io.host_block_table = lambda *, layer_id: host_table
    batch = build_dsa_offload_batch(
        layout=layout,
        hot_cache=hot_cache,
        io_backend=spy_io,
        cohorts=(cohort,),
        lookup_states={},
        request_ids=("prefill", "decode"),
        query_counts=(2, 1),
        query_positions=torch.tensor([0, 1, 5], dtype=torch.int64),
        query_positions_cpu=[0, 1, 5],
        is_mtp=False,
        enable_cohort_kvgather=enable_cohort_kvgather,
        fuse_kvgather_sfa=True,
        fused_staging_cache=(torch.empty(1), torch.empty(1)),
        committed_block_hashes={"prefill": [], "decode": [b"\x01" * 32]},
        candidate_block_hashes={},
        sfa_workspace=SFAAddressingWorkspace.create(
            max_num_seqs=layout.max_num_seqs,
            max_block_table_width=layout.hot_blocks_per_row,
            device="cpu",
        ),
    )
    return batch, row, host_key, host_table


def run_cohort(batch, plan, mapped):
    table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    seq_lens = torch.tensor([2, 6], dtype=torch.int32)
    events = []

    def make_plan(**kwargs):
        events.append("lookup")
        return plan

    def load_misses(plan, layer_id, batch):
        events.append(f"get:{layer_id}")

    with (
        patch("vllm_ascend.dsa_offload.lookup.make_lookup_plan", side_effect=make_plan),
        patch("vllm_ascend.dsa_offload.lookup.load_plan_misses", side_effect=load_misses),
    ):
        leader = resolve_sfa_inputs(
            layer_name="leader",
            semantic_topk=mapped,
            default_block_table=table,
            default_actual_seq_lengths_kv=seq_lens,
            batch=batch,
        )
        events.append("sfa:leader")
        follower = resolve_sfa_inputs(
            layer_name="follower",
            semantic_topk=mapped,
            default_block_table=table,
            default_actual_seq_lengths_kv=seq_lens,
            batch=batch,
        )
        events.append("sfa:follower")
    return leader, follower, events


def make_plan_with_route() -> tuple[LookupPlan, torch.Tensor]:
    mapped = torch.tensor([[8, 9]], dtype=torch.int32)
    empty = torch.empty(0, dtype=torch.int64)
    route_plan = torch.ones((3, 2048), dtype=torch.int32)
    plan = LookupPlan(
        mapped_indices=mapped,
        miss_positions=empty,
        miss_logical_blocks=empty,
        miss_block_offsets=empty,
        miss_destination_slots=empty,
        miss_batch_indices=empty.to(torch.int32),
        query_request_rows=empty.to(torch.int32),
        route_plan=route_plan,
    )
    return plan, mapped


def test_cohort_leader_fuses_and_skips_own_gather(spy_io) -> None:
    batch, row, host_key, host_table = make_fused_batch(
        spy_io, enable_cohort_kvgather=True
    )
    plan, mapped = make_plan_with_route()

    leader, follower, events = run_cohort(batch, plan, mapped)

    # The leader's standalone gather is absorbed by the fused operator while
    # the follower gather keeps its early issue at cohort start.
    assert events == ["lookup", "get:4", "sfa:leader", "sfa:follower"]
    assert leader.fused is not None
    assert follower.fused is None
    assert leader.sparse_indices is mapped
    assert leader.actual_seq_lengths_kv.tolist() == [2, 6]
    assert torch.equal(leader.block_table, batch.hot_cache.hot_block_table)
    assert leader.fused.route_plan is plan.route_plan
    assert leader.fused.host_key is host_key
    assert leader.fused.host_block_table is host_table
    # The fused operator pads the per-request rows with -1 up to the query-row
    # count of the route plan (MTP requests expand to multiple query rows).
    packed_rows = batch.packed_decode.request_rows
    assert torch.equal(
        leader.fused.request_rows[: packed_rows.shape[0]],
        packed_rows,
    )
    assert (leader.fused.request_rows[packed_rows.shape[0]:] == -1).all()
    # The follower keeps the ordinary remapped addressing view.
    assert follower.sparse_indices is plan.mapped_indices
    assert follower.actual_seq_lengths_kv.tolist() == [2, batch.layout.row_stride]


def test_all_layers_fuse_when_cohort_kvgather_disabled(spy_io) -> None:
    batch, _, _, _ = make_fused_batch(spy_io, enable_cohort_kvgather=False)
    plan, mapped = make_plan_with_route()

    leader, follower, events = run_cohort(batch, plan, mapped)

    assert events == ["lookup", "sfa:leader", "sfa:follower"]
    assert leader.fused is not None
    assert follower.fused is not None


def test_backend_without_host_planes_falls_back(spy_io) -> None:
    batch, _, _, _ = make_fused_batch(
        spy_io, enable_cohort_kvgather=True, with_host_planes=False
    )
    plan, mapped = make_plan_with_route()

    leader, follower, events = run_cohort(batch, plan, mapped)

    assert leader.fused is None
    assert follower.fused is None
    assert events == ["lookup", "get:3", "get:4", "sfa:leader", "sfa:follower"]


def test_missing_staging_buffer_falls_back(spy_io) -> None:
    batch, _, _, _ = make_fused_batch(spy_io, enable_cohort_kvgather=False)
    batch.fused_staging_cache = None
    plan, mapped = make_plan_with_route()

    leader, _, events = run_cohort(batch, plan, mapped)

    assert leader.fused is None
    assert events[1] == "get:3"
