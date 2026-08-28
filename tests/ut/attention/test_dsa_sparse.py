# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.attention.dsa_sparse import (
    INVALID_INDEX,
    CacheSeatManager,
    DSASparseCacheConfig,
    DSASparseCohortKey,
    DSASparseLayerHotCache,
    DSASparseLayerLayout,
    DSASparseBatchMetadata,
    DSASparsePlan,
    DSASparsePlanKey,
    DSASparseResidencyState,
    UnimplementedDSASparseLookupUpdateOperator,
    dsa_sparse_lookup_workspace_stride,
)


def make_cache_config() -> DSASparseCacheConfig:
    return DSASparseCacheConfig(
        max_num_seqs=3,
        max_model_len=128,
        block_size=16,
        device_buffer_size=32,
        max_query_tokens_per_request=4,
        index_topk=8,
    )


def test_cache_config_accounts_for_reserved_and_aligned_slots():
    config = make_cache_config()

    assert config.max_topk_union_width == 32
    assert config.evictable_hot_slots == range(0, 32)
    assert config.reserved_newest_slots == range(32, 36)
    assert config.alignment_padding_slots == range(36, 48)
    assert config.managed_hot_width == 36
    assert config.hot_stride == 48
    assert config.hot_blocks_per_seat == 3
    assert config.total_hot_blocks == 9


def test_cache_config_rejects_topk_union_larger_than_evictable_cache():
    with pytest.raises(ValueError, match="complete per-request Top-K union"):
        DSASparseCacheConfig(
            max_num_seqs=3,
            max_model_len=128,
            block_size=16,
            device_buffer_size=31,
            max_query_tokens_per_request=4,
            index_topk=8,
        )


def test_cache_config_rejects_more_than_four_query_lanes():
    with pytest.raises(ValueError, match="operator limit of 4"):
        DSASparseCacheConfig(
            max_num_seqs=3,
            max_model_len=128,
            block_size=16,
            device_buffer_size=40,
            max_query_tokens_per_request=5,
            index_topk=8,
        )


def test_cache_seat_manager_keeps_seat_stable_across_row_reorder():
    manager = CacheSeatManager(max_num_seqs=3)
    first = manager.acquire("request-a")
    second = manager.acquire("request-b")
    config = make_cache_config()
    plan = DSASparsePlan.allocate(
        config,
        DSASparsePlanKey(
            token_capacity=3,
            request_capacity=3,
            query_lane_capacity=1,
            role="target",
        ),
        device="cpu",
    )

    original = manager.pack_rows(
        ["request-a", "request-b"],
        plan.row_mapping,
    )
    original_seats = original.row_to_cache_seat.clone()
    reordered = manager.pack_rows(
        ["request-b", "request-a"],
        plan.row_mapping,
    )

    assert original is plan.row_mapping
    assert reordered is plan.row_mapping
    assert original_seats.tolist() == [
        first.seat,
        second.seat,
        INVALID_INDEX,
    ]
    assert reordered.row_to_cache_seat.tolist() == [
        second.seat,
        first.seat,
        INVALID_INDEX,
    ]
    assert reordered.row_seat_epoch.tolist() == [
        second.epoch,
        first.epoch,
        INVALID_INDEX,
    ]


def test_cache_seat_reuse_increments_epoch():
    manager = CacheSeatManager(max_num_seqs=1)
    previous = manager.acquire("request-a")
    manager.release("request-a")
    current = manager.acquire("request-b")

    assert current.seat == previous.seat
    assert current.epoch == previous.epoch + 1


def test_cache_seat_manager_rejects_duplicate_active_rows():
    manager = CacheSeatManager(max_num_seqs=2)
    manager.acquire("request-a")
    config = make_cache_config()
    plan = DSASparsePlan.allocate(
        config,
        DSASparsePlanKey(
            token_capacity=2,
            request_capacity=2,
            query_lane_capacity=1,
            role="target",
        ),
        device="cpu",
    )

    with pytest.raises(ValueError, match="only one DSA Sparse row"):
        manager.pack_rows(
            ["request-a", "request-a"],
            plan.row_mapping,
        )


def test_residency_state_excludes_reserved_slots_from_lru():
    config = make_cache_config()
    cohort = DSASparseCohortKey(name="layers.0-3", role="target")
    state = DSASparseResidencyState.allocate(
        config,
        cohort,
        device="cpu",
    )

    assert state.cohort is cohort
    assert state.token_to_hot.shape == (3, 128)
    assert state.hot_to_token.shape == (3, 32)
    assert state.lru_slots.shape == (3, 32)
    assert state.state_seat_epoch.shape == (3,)
    assert state.lru_slots[0].tolist() == list(range(32))
    assert torch.all(state.token_to_hot == INVALID_INDEX)
    assert torch.all(state.hot_to_token == INVALID_INDEX)


def test_plan_uses_fixed_shapes_and_hot_block_stride():
    config = make_cache_config()
    key = DSASparsePlanKey(
        token_capacity=8,
        request_capacity=2,
        query_lane_capacity=4,
        role="target",
    )
    plan = DSASparsePlan.allocate(config, key, device="cpu")

    assert plan.resolved_hot_indices.shape == (8, 8)
    assert plan.miss_mask.shape == (8, 8)
    assert plan.workspace.shape == (
        2,
        dsa_sparse_lookup_workspace_stride(32),
    )
    assert plan.hot_block_table.shape == (2, 3)
    assert plan.row_mapping.row_to_cache_seat.shape == (2,)
    assert plan.query_positions.shape == (8,)
    assert plan.query_to_row.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert plan.query_to_lane.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert plan.query_valid_mask.shape == (8,)
    assert plan.valid_topk_counts.shape == (8,)
    assert plan.topk_positions.shape == (8, 8)
    assert plan.seq_lens.shape == (2,)
    assert plan.block_table.shape == (2, 8)
    assert plan.write_global_slots.shape == (2, 4)
    assert plan.write_destination_hot_row_ids.shape == (2, 4)
    assert plan.write_valid_mask.shape == (2, 4)


@pytest.mark.parametrize(
    ("plane_dtypes", "plane_row_shapes", "expected_shapes"),
    [
        (
            (torch.bfloat16, torch.bfloat16),
            ((1, 512), (1, 64)),
            ((9, 16, 1, 512), (9, 16, 1, 64)),
        ),
        (
            (torch.int8,),
            ((1, 704),),
            ((9, 16, 1, 704),),
        ),
    ],
)
def test_layer_hot_cache_preserves_existing_sfa_plane_layout(
    plane_dtypes,
    plane_row_shapes,
    expected_shapes,
):
    config = make_cache_config()
    layout = DSASparseLayerLayout(
        layer_name="model.layers.0.self_attn",
        plane_dtypes=plane_dtypes,
        plane_row_shapes=plane_row_shapes,
    )

    hot_cache = DSASparseLayerHotCache.allocate(
        layout,
        config,
        device="cpu",
    )

    assert tuple(plane.shape for plane in hot_cache.planes) == expected_shapes
    assert tuple(plane.dtype for plane in hot_cache.planes) == plane_dtypes


def test_unimplemented_lookup_update_operator_is_an_explicit_stub():
    operator = UnimplementedDSASparseLookupUpdateOperator()

    with pytest.raises(NotImplementedError, match="lookup/update"):
        operator.lookup_update()


def test_plans_can_share_role_level_batch_metadata():
    config = make_cache_config()
    key = DSASparsePlanKey(
        token_capacity=8,
        request_capacity=2,
        query_lane_capacity=4,
        role="target",
    )
    metadata = DSASparseBatchMetadata.allocate(
        config,
        key,
        device="cpu",
    )
    first = DSASparsePlan.allocate(
        config,
        key,
        device="cpu",
        batch_metadata=metadata,
    )
    second = DSASparsePlan.allocate(
        config,
        key,
        device="cpu",
        batch_metadata=metadata,
    )

    assert first.batch_metadata is second.batch_metadata
    assert first.topk_positions.data_ptr() != second.topk_positions.data_ptr()
    assert first.workspace.data_ptr() != second.workspace.data_ptr()


def test_plan_key_rejects_non_rectangular_query_mapping():
    with pytest.raises(ValueError, match="token_capacity must equal"):
        DSASparsePlanKey(
            token_capacity=7,
            request_capacity=2,
            query_lane_capacity=4,
            role="target",
        )


def test_cohort_role_is_limited_to_target_and_draft():
    with pytest.raises(ValueError, match="target.*draft"):
        DSASparseCohortKey(name="layers.0-3", role="invalid")
