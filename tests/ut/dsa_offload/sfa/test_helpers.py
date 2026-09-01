# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import Mock, patch

import torch

from vllm_ascend.dsa_offload.hot_cache import HotCacheLayout, HotCacheState
from vllm_ascend.dsa_offload.lookup import (
    DSAOffloadBatch,
    IndexCacheCohort,
    LookupPlan,
    build_dsa_offload_batch,
)
from vllm_ascend.dsa_offload.sfa import (
    SFAAddressingWorkspace,
    prepare_indexer_cache_write,
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
        query_start_loc=torch.tensor(
            [0, 2, 4 if is_mtp else 3], dtype=torch.int32
        ),
        query_positions=torch.tensor(
            [0, 1, 5, 6] if is_mtp else [0, 1, 5],
            dtype=torch.int64,
        ),
        query_positions_cpu=(0, 1, 5, 6)
        if is_mtp
        else (0, 1, 5),
        is_mtp=is_mtp,
        committed_block_keys={"prefill": [], "decode": [101]},
        candidate_block_keys={},
        sfa_workspace=SFAAddressingWorkspace.create(
            max_num_seqs=layout.max_num_seqs,
            max_block_table_width=layout.hot_blocks_per_row,
            device="cpu",
        ),
    )
    return batch, row


def test_feature_off_returns_original_inputs() -> None:
    slots = torch.tensor([1, 2], dtype=torch.int64)
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    table = torch.tensor([[3, 4]], dtype=torch.int32)
    seq_lens = torch.tensor([2], dtype=torch.int32)

    assert prepare_main_slot_mapping(batch=None, default_slot_mapping=slots) is slots
    resolved = resolve_sfa_inputs(
        layer_name="layer",
        semantic_topk=topk,
        default_block_table=table,
        default_actual_seq_lengths_kv=seq_lens,
        batch=None,
    )
    assert resolved.sparse_indices is topk
    assert resolved.block_table is table
    assert resolved.actual_seq_lengths_kv is seq_lens


def test_prefetch_target_key_write_updates_mean_cache() -> None:
    runtime = Mock(target_layer_names=frozenset({"target"}))
    batch = DSAOffloadBatch(
        layout=HotCacheLayout(4, 1, 2),
        hot_cache=None,
        io_backend=Mock(),
        cohorts=(),
        lookup_states={},
        request_ids=("decode",),
        request_rows=torch.tensor([0], dtype=torch.int32),
        request_rows_cpu=(0,),
        decode_request_indices=(0,),
        query_ranges=((0, 1),),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_positions=torch.tensor([8], dtype=torch.int64),
        query_positions_cpu=(8,),
        is_mtp=False,
        committed_block_keys={"decode": []},
        candidate_block_keys={},
        prefetch_runtime=runtime,
    )
    key_cache = torch.empty((2, 4, 1, 8))
    key_mean = torch.empty((2, 1, 1, 8))
    key = torch.empty((1, 1, 8))
    slot_mapping = torch.tensor([0], dtype=torch.int64)

    with patch.object(
        torch.ops._C_ascend,
        "npu_scatter_nd_update_mean",
        create=True,
    ) as update_mean:
        handled = prepare_indexer_cache_write(
            layer_name="target",
            key_cache=key_cache,
            indexer_cache=(key_cache, key_mean),
            enable_sparse_li_c8=False,
            slot_mapping=slot_mapping,
            key=key,
            block_size=4,
            batch=batch,
        )

    assert handled
    runtime.wait_for_compute_before_key_write.assert_called_once_with("target")
    update_mean.assert_called_once()
    assert update_mean.call_args.args[3] is key_mean


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


def test_graph_mtp_mapping_uses_runtime_request_rows(spy_io) -> None:
    layout = HotCacheLayout(4, 2, 2)
    batch = DSAOffloadBatch(
        layout=layout,
        hot_cache=None,
        io_backend=spy_io,
        cohorts=(),
        lookup_states={},
        request_ids=("first", "second"),
        request_rows=torch.tensor([1, 0], dtype=torch.int32),
        request_rows_cpu=(1, 0),
        decode_request_indices=(0, 1),
        query_ranges=((0, 2), (2, 3)),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        query_positions=torch.tensor([8, 9, 12], dtype=torch.int64),
        query_positions_cpu=(8, 9, 12),
        is_mtp=True,
        committed_block_keys={"first": [], "second": []},
        candidate_block_keys={},
        graph_query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
    )

    mapped = prepare_main_slot_mapping(
        batch=batch,
        default_slot_mapping=torch.full((3,), -1, dtype=torch.int64),
    )

    assert mapped.tolist() == [
        layout.global_slot(1, layout.staging_base),
        layout.global_slot(1, layout.staging_base + 1),
        layout.global_slot(0, layout.staging_base),
    ]


def test_leader_looks_up_once_and_follower_performs_own_get(spy_io) -> None:
    batch, row = make_mixed_batch(spy_io)
    assert batch.hot_cache is not None
    mapped = torch.tensor([[8, 9]], dtype=torch.int32)
    table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    seq_lens = torch.tensor([2, 6], dtype=torch.int32)
    empty = torch.empty(0, dtype=torch.int64)
    plan = LookupPlan(
        mapped_indices=mapped,
        miss_positions=empty,
        miss_logical_blocks=empty,
        miss_block_offsets=empty,
        miss_destination_slots=empty,
        miss_batch_indices=empty.to(torch.int32),
        query_request_rows=empty.to(torch.int32),
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
        leader_addressing = resolve_sfa_inputs(
            layer_name="leader",
            semantic_topk=mapped,
            default_block_table=table,
            default_actual_seq_lengths_kv=seq_lens,
            batch=batch,
        )
        assert leader_addressing.sparse_indices is mapped
        assert leader_addressing.block_table[0, :2].tolist() == [1, 2]
        assert torch.count_nonzero(leader_addressing.block_table[0, 2:]) == 0
        assert torch.equal(
            leader_addressing.block_table[1],
            batch.hot_cache.hot_block_table[row],
        )
        assert leader_addressing.actual_seq_lengths_kv.tolist() == [2, batch.layout.row_stride]
        events.append("sfa:leader")
        follower_default = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
        follower_addressing = resolve_sfa_inputs(
            layer_name="follower",
            semantic_topk=mapped,
            default_block_table=follower_default,
            default_actual_seq_lengths_kv=seq_lens,
            batch=batch,
        )
        assert follower_addressing.sparse_indices is mapped
        assert follower_addressing.block_table[0, :2].tolist() == [5, 6]
        assert torch.count_nonzero(follower_addressing.block_table[0, 2:]) == 0
        assert torch.equal(
            follower_addressing.block_table[1],
            batch.hot_cache.hot_block_table[row],
        )
        assert follower_addressing.actual_seq_lengths_kv.tolist() == [2, batch.layout.row_stride]
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


def test_fixed_hot_addressing_does_not_depend_on_model_block_table_width() -> None:
    layout = HotCacheLayout(128, 2, 16)
    cache = torch.empty((layout.hot_blocks, layout.block_size, 1))
    hot_cache = HotCacheState(layout, {"layer": (cache,)})
    row = hot_cache.admit("decode")
    workspace = SFAAddressingWorkspace.create(
        max_num_seqs=layout.max_num_seqs,
        max_block_table_width=layout.hot_blocks_per_row,
        device="cpu",
    )
    batch = build_dsa_offload_batch(
        layout=layout,
        hot_cache=hot_cache,
        io_backend=Mock(),
        cohorts=(),
        lookup_states={},
        request_ids=("prefill", "decode"),
        query_counts=(1, 1),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_positions=torch.tensor([0, 4095], dtype=torch.int64),
        query_positions_cpu=(0, 4095),
        is_mtp=False,
        committed_block_keys={"prefill": [], "decode": []},
        candidate_block_keys={},
        sfa_workspace=workspace,
    )
    ordinary_table = torch.arange(64, dtype=torch.int32).reshape(2, 32)
    ordinary_seq_lens = torch.tensor([128, 4096], dtype=torch.int32)

    effective_table, effective_seq_lens = workspace.compose(
        default_block_table=ordinary_table,
        default_actual_seq_lengths_kv=ordinary_seq_lens,
        batch=batch,
    )

    assert effective_table.shape == (2, 82)
    assert effective_table[0, :32].tolist() == ordinary_table[0].tolist()
    assert torch.count_nonzero(effective_table[0, 32:]) == 0
    assert torch.equal(effective_table[1], hot_cache.hot_block_table[row])
    assert effective_seq_lens.tolist() == [128, 10496]
    assert ordinary_table.shape == (2, 32)
    assert ordinary_seq_lens.tolist() == [128, 4096]
