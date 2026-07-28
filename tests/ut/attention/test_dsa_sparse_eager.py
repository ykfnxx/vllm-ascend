# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import FrozenInstanceError, dataclass, replace

import pytest
import torch

from vllm_ascend.attention.dsa_sparse import (
    DSASparseCacheConfig,
    DSASparseCohort,
    DSASparseCohortKey,
    DSASparseEagerBatchContext,
    DSASparseEagerContextRouter,
    DSASparseEagerCoordinator,
    DSASparseLayerBinding,
    DSASparseLayerHotCache,
    DSASparseLayerLayout,
    DSASparsePlan,
    DSASparsePlanKey,
    DSASparseResidencyState,
    DSASparseResolution,
    RequestIndexManager,
)


@dataclass
class RecordingIndexOperator:
    events: list[str]
    has_misses: bool

    def lookup_update(self, *, state, plan):
        del state
        self.events.append("lookup_update")
        plan.resolved_hot_indices.zero_()
        plan.miss_mask.zero_()
        if self.has_misses:
            plan.miss_mask[0, 0] = True


@dataclass
class FailingIndexOperator(RecordingIndexOperator):
    def lookup_update(self, **kwargs):
        super().lookup_update(**kwargs)
        raise RuntimeError("lookup/update failed")


@dataclass
class RecordingIOOperator:
    events: list[str]
    transfer_counts: list[int]

    def dsa_sparse_io(
        self,
        *,
        context,
        region,
        topk_positions,
        resolved_hot_indices,
        miss_mask,
        query_to_req_idx,
        block_table,
        write_global_slots,
        write_destination_hot_row_ids,
        write_valid_mask,
        hot_planes,
        completion,
    ):
        del (
            context,
            topk_positions,
            resolved_hot_indices,
            query_to_req_idx,
            block_table,
            write_global_slots,
            write_destination_hot_row_ids,
            write_valid_mask,
            hot_planes,
            completion,
        )
        self.events.append(f"dsa_sparse_io:{region}")
        self.transfer_counts.append(int(miss_mask.sum()))


@dataclass
class FailingIOOperator(RecordingIOOperator):
    def dsa_sparse_io(self, **kwargs):
        super().dsa_sparse_io(**kwargs)
        raise RuntimeError("unified I/O completion failed")


@dataclass
class FailingOnceIOOperator(RecordingIOOperator):
    has_failed: bool = False

    def dsa_sparse_io(self, **kwargs):
        super().dsa_sparse_io(**kwargs)
        if not self.has_failed:
            self.has_failed = True
            raise RuntimeError("unified I/O completion failed once")


def build_coordinator(
    *,
    has_misses: bool,
    layer_names: tuple[str, ...] = ("layer.0", "layer.1"),
    freeze: bool = True,
) -> tuple[
    DSASparseEagerCoordinator,
    DSASparseCohortKey,
    DSASparsePlanKey,
    list[str],
    list[int],
]:
    config = DSASparseCacheConfig(
        max_num_seqs=2,
        max_model_len=32,
        block_size=4,
        device_buffer_size=8,
        max_query_tokens_per_request=2,
        index_topk=4,
    )
    cohort_key = DSASparseCohortKey(
        name="shared-indexer-0",
        role="target",
    )
    plan_key = DSASparsePlanKey(
        token_capacity=4,
        request_capacity=2,
        query_lane_capacity=2,
        role="target",
    )
    state = DSASparseResidencyState.allocate(
        config,
        cohort_key,
        device="cpu",
    )
    plan = DSASparsePlan.allocate(
        config,
        plan_key,
        device="cpu",
    )
    cohort = DSASparseCohort(
        key=cohort_key,
        leader_layer="layer.0",
        state=state,
        plans={plan_key: plan},
    )
    events: list[str] = []
    transfer_counts: list[int] = []
    coordinator = DSASparseEagerCoordinator(
        config,
        index_operator=RecordingIndexOperator(events, has_misses),
        io_operator=RecordingIOOperator(events, transfer_counts),
        request_index_manager=RequestIndexManager(config.max_num_seqs),
    )
    coordinator.register_cohort(cohort)
    layout = DSASparseLayerLayout(
        layer_name="unused",
        plane_dtypes=(torch.bfloat16, torch.bfloat16),
        plane_row_shapes=((1, 8), (1, 2)),
    )
    for layer_name in layer_names:
        layer_layout = DSASparseLayerLayout(
            layer_name=layer_name,
            plane_dtypes=layout.plane_dtypes,
            plane_row_shapes=layout.plane_row_shapes,
        )
        hot_cache = DSASparseLayerHotCache.allocate(
            layer_layout,
            config,
            device="cpu",
        )
        coordinator.register_layer(
            DSASparseLayerBinding(
                layer_name=layer_name,
                cohort=cohort_key,
                hot_cache=hot_cache,
                io_context=f"context:{layer_name}",
                io_region=f"region:{layer_name}",
                io_completion=object(),
            )
        )
    coordinator.acquire_request("request-a")
    coordinator.acquire_request("request-b")
    if freeze:
        coordinator.freeze()
    return coordinator, cohort_key, plan_key, events, transfer_counts


def begin_step(
    coordinator: DSASparseEagerCoordinator,
    cohort_key: DSASparseCohortKey,
    plan_key: DSASparsePlanKey,
):
    return coordinator.begin_step(
        cohort_key,
        plan_key,
        request_ids=["request-a", "request-b"],
        request_indices=[0, 1],
        query_positions=torch.tensor([5, 6, 9, -1], dtype=torch.int32),
        query_valid_mask=torch.tensor(
            [True, True, True, False],
            dtype=torch.bool,
        ),
        seq_lens=torch.tensor([7, 10], dtype=torch.int32),
        block_table=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
            ],
            dtype=torch.int32,
        ),
    )


def test_begin_step_builds_hot_blocks_and_newest_descriptors():
    coordinator, cohort_key, plan_key, _events, _counts = (
        build_coordinator(
            has_misses=False,
            layer_names=("layer.0",),
        )
    )

    step = begin_step(coordinator, cohort_key, plan_key)

    assert step.plan.hot_block_table.tolist() == [
        [0, 1, 2],
        [3, 4, 5],
    ]
    assert step.plan.write_global_slots.tolist() == [
        [5, 6],
        [41, -1],
    ]
    assert step.plan.write_destination_hot_row_ids.tolist() == [
        [8, 9],
        [20, -1],
    ]
    assert step.plan.write_valid_mask.tolist() == [
        [True, True],
        [True, False],
    ]
    coordinator.abort_step(step)


@pytest.mark.parametrize(
    ("has_misses", "expected_transfer_count"),
    [(False, 0), (True, 1)],
)
def test_eager_coordinator_keeps_one_fixed_flow_for_hits_and_misses(
    has_misses,
    expected_transfer_count,
):
    (
        coordinator,
        cohort_key,
        plan_key,
        events,
        transfer_counts,
    ) = build_coordinator(
        has_misses=has_misses,
        layer_names=("layer.0",),
    )
    step = begin_step(coordinator, cohort_key, plan_key)
    coordinator.submit_newest_write(step, "layer.0")
    coordinator.prepare_lookup(
        step,
        topk_positions=torch.zeros((4, 4), dtype=torch.int32),
        valid_topk_counts=torch.tensor(
            [4, 4, 4, 0],
            dtype=torch.int32,
        ),
    )

    def existing_sfa(resolution: DSASparseResolution):
        events.append("existing_sfa")
        assert resolution.hot_main_cache[0].shape == (6, 4, 1, 8)
        assert resolution.local_sparse_indices is step.plan.resolved_hot_indices
        assert resolution.hot_block_table is step.plan.hot_block_table
        return torch.tensor([123])

    output = coordinator.run_layer_attention(
        step,
        "layer.0",
        existing_sfa,
    )
    coordinator.finish_step(step)

    assert output.tolist() == [123]
    assert events == [
        "lookup_update",
        "dsa_sparse_io:region:layer.0",
        "existing_sfa",
    ]
    assert transfer_counts == [expected_transfer_count]


def test_cohort_followers_reuse_lookup_but_keep_layer_io_resources():
    (
        coordinator,
        cohort_key,
        plan_key,
        events,
        transfer_counts,
    ) = build_coordinator(has_misses=False)
    step = begin_step(coordinator, cohort_key, plan_key)
    coordinator.submit_newest_write(step, "layer.0")
    coordinator.prepare_lookup(
        step,
        topk_positions=torch.zeros((4, 4), dtype=torch.int32),
        valid_topk_counts=torch.tensor(
            [4, 4, 4, 0],
            dtype=torch.int32,
        ),
    )
    coordinator.run_layer_attention(
        step,
        "layer.0",
        lambda resolution: torch.tensor([resolution.hot_main_cache[0].shape[0]]),
    )
    coordinator.submit_newest_write(step, "layer.1")
    coordinator.run_layer_attention(
        step,
        "layer.1",
        lambda resolution: torch.tensor([resolution.hot_main_cache[0].shape[0]]),
    )
    coordinator.finish_step(step)

    assert events.count("lookup_update") == 1
    assert "dsa_sparse_io:region:layer.0" in events
    assert "dsa_sparse_io:region:layer.1" in events
    assert transfer_counts == [0, 0]


def test_step_cannot_finish_before_all_layer_writes_join():
    (
        coordinator,
        cohort_key,
        plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(has_misses=False)
    step = begin_step(coordinator, cohort_key, plan_key)

    with pytest.raises(RuntimeError, match="pending layers"):
        coordinator.finish_step(step)


def test_same_plan_cannot_be_reused_while_step_is_active():
    (
        coordinator,
        cohort_key,
        plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(has_misses=False)
    begin_step(coordinator, cohort_key, plan_key)

    with pytest.raises(RuntimeError, match="one DSA Sparse step"):
        begin_step(coordinator, cohort_key, plan_key)


def test_same_cohort_cannot_run_two_plan_shapes_concurrently():
    (
        coordinator,
        cohort_key,
        plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(has_misses=False, freeze=False)
    second_key = DSASparsePlanKey(
        token_capacity=2,
        request_capacity=2,
        query_lane_capacity=1,
        role="target",
    )
    cohort = coordinator.get_cohort(cohort_key)
    cohort.plans[second_key] = DSASparsePlan.allocate(
        coordinator.config,
        second_key,
        device="cpu",
    )
    coordinator.freeze()
    begin_step(coordinator, cohort_key, plan_key)

    with pytest.raises(RuntimeError, match="one DSA Sparse step"):
        coordinator.begin_step(
            cohort_key,
            second_key,
            request_ids=["request-a", "request-b"],
            request_indices=[0, 1],
            query_positions=torch.tensor([5, 9], dtype=torch.int32),
            query_valid_mask=torch.tensor([True, True]),
            seq_lens=torch.tensor([7, 10], dtype=torch.int32),
            block_table=torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [8, 9, 10, 11, 12, 13, 14, 15],
                ],
                dtype=torch.int32,
            ),
        )


def test_lookup_requires_leader_newest_write():
    (
        coordinator,
        cohort_key,
        plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(has_misses=False)
    step = begin_step(coordinator, cohort_key, plan_key)

    with pytest.raises(RuntimeError, match="leader.*newest Main KV write"):
        coordinator.prepare_lookup(
            step,
            topk_positions=torch.zeros((4, 4), dtype=torch.int32),
            valid_topk_counts=torch.tensor(
                [4, 4, 4, 0],
                dtype=torch.int32,
            ),
        )


def test_request_release_waits_for_active_step_to_finish():
    (
        coordinator,
        cohort_key,
        plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    step = begin_step(coordinator, cohort_key, plan_key)

    with pytest.raises(RuntimeError, match="pending layer I/O"):
        coordinator.release_request("request-a")

    coordinator.submit_newest_write(step, "layer.0")
    coordinator.prepare_lookup(
        step,
        topk_positions=torch.zeros((4, 4), dtype=torch.int32),
        valid_topk_counts=torch.tensor(
            [4, 4, 4, 0],
            dtype=torch.int32,
        ),
    )
    coordinator.run_layer_attention(
        step,
        "layer.0",
        lambda resolution: torch.tensor([resolution.hot_main_cache[0].shape[0]]),
    )
    coordinator.finish_step(step)

    with pytest.raises(RuntimeError, match="active cohort owner"):
        coordinator.submit_newest_write(step, "layer.0")

    coordinator.release_request("request-a")


def test_reused_request_index_clears_previous_residency_state():
    (
        coordinator,
        cohort_key,
        _plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    state = coordinator.get_cohort(cohort_key).state
    state.token_to_hot[0, 7] = 3
    state.hot_to_token[0, 3] = 7
    state.lru_slots[0].copy_(state.lru_slots[0].flip(0))

    released_index = coordinator.release_request("request-a")
    reused_index = coordinator.acquire_request("request-c")

    assert reused_index == released_index == 0
    assert torch.all(state.token_to_hot[0] == -1)
    assert torch.all(state.hot_to_token[0] == -1)
    assert state.lru_slots[0].tolist() == list(
        range(state.lru_slots.shape[1])
    )


def test_coordinator_rejects_plan_that_cannot_address_full_request_pool():
    config = DSASparseCacheConfig(
        max_num_seqs=2,
        max_model_len=32,
        block_size=4,
        device_buffer_size=8,
        max_query_tokens_per_request=2,
        index_topk=4,
    )
    cohort_key = DSASparseCohortKey(
        name="shared-indexer-0",
        role="target",
    )
    undersized_plan_key = DSASparsePlanKey(
        token_capacity=2,
        request_capacity=1,
        query_lane_capacity=2,
        role="target",
    )
    coordinator = DSASparseEagerCoordinator(
        config,
        index_operator=RecordingIndexOperator([], False),
        io_operator=RecordingIOOperator([], []),
    )
    cohort = DSASparseCohort(
        key=cohort_key,
        leader_layer="layer.0",
        state=DSASparseResidencyState.allocate(
            config,
            cohort_key,
            device="cpu",
        ),
        plans={
            undersized_plan_key: DSASparsePlan.allocate(
                config,
                undersized_plan_key,
                device="cpu",
            )
        },
    )

    with pytest.raises(ValueError, match="complete request-index pool"):
        coordinator.register_cohort(cohort)


def test_layer_registration_rejects_aliased_hot_cache():
    (
        coordinator,
        cohort_key,
        _plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
        freeze=False,
    )
    leader_binding = coordinator.get_layer_binding(cohort_key, "layer.0")
    aliased_hot_cache = DSASparseLayerHotCache(
        layer_name="layer.1",
        planes=leader_binding.hot_cache.planes,
    )

    with pytest.raises(ValueError, match="independent Hot Cache"):
        coordinator.register_layer(
            DSASparseLayerBinding(
                layer_name="layer.1",
                cohort=cohort_key,
                hot_cache=aliased_hot_cache,
                io_context=object(),
                io_region=object(),
                io_completion=object(),
            )
        )


def test_target_and_draft_may_register_the_same_layer_name():
    (
        coordinator,
        target_key,
        _plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
        freeze=False,
    )
    config = coordinator.config
    draft_key = DSASparseCohortKey(
        name="shared-indexer-0",
        role="draft",
    )
    draft_plan_key = DSASparsePlanKey(
        token_capacity=4,
        request_capacity=2,
        query_lane_capacity=2,
        role="draft",
    )
    coordinator.register_cohort(
        DSASparseCohort(
            key=draft_key,
            leader_layer="layer.0",
            state=DSASparseResidencyState.allocate(
                config,
                draft_key,
                device="cpu",
            ),
            plans={
                draft_plan_key: DSASparsePlan.allocate(
                    config,
                    draft_plan_key,
                    device="cpu",
                )
            },
        )
    )
    draft_hot_cache = DSASparseLayerHotCache.allocate(
        DSASparseLayerLayout(
            layer_name="layer.0",
            plane_dtypes=(torch.bfloat16, torch.bfloat16),
            plane_row_shapes=((1, 8), (1, 2)),
        ),
        config,
        device="cpu",
    )
    coordinator.register_layer(
        DSASparseLayerBinding(
            layer_name="layer.0",
            cohort=draft_key,
            hot_cache=draft_hot_cache,
            io_context="context:draft:layer.0",
            io_region="region:draft:layer.0",
            io_completion=object(),
        )
    )
    coordinator.freeze()

    target_binding = coordinator.get_layer_binding(target_key, "layer.0")
    draft_binding = coordinator.get_layer_binding(draft_key, "layer.0")
    assert target_binding is not draft_binding
    assert target_binding.hot_cache.planes[0].data_ptr() != draft_binding.hot_cache.planes[0].data_ptr()


def test_freeze_keeps_cohort_metadata_and_plan_map_immutable():
    (
        coordinator,
        cohort_key,
        plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    cohort = coordinator.get_cohort(cohort_key)

    with pytest.raises(FrozenInstanceError):
        cohort.leader_layer = "layer.1"
    with pytest.raises(TypeError):
        cohort.plans[plan_key] = DSASparsePlan.allocate(
            coordinator.config,
            plan_key,
            device="cpu",
        )


def begin_dynamic_batch(
    coordinator,
    cohort_key,
    plan_key,
    *,
    num_sfa_queries=None,
    query_positions_dtype: torch.dtype = torch.int32,
):
    return DSASparseEagerBatchContext.begin(
        coordinator,
        cohort_key,
        plan_key,
        request_ids=["request-a", "request-b"],
        request_indices=[0, 1],
        query_positions=torch.tensor(
            [5, 9],
            dtype=query_positions_dtype,
        ),
        query_counts=[1, 1],
        seq_lens=torch.tensor([7, 10], dtype=torch.int32),
        block_table=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
            ],
            dtype=torch.int32,
        ),
        num_sfa_queries=num_sfa_queries,
    )


def test_dynamic_eager_batch_packs_lanes_and_returns_active_sfa_view():
    (
        coordinator,
        cohort_key,
        plan_key,
        events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    context = begin_dynamic_batch(coordinator, cohort_key, plan_key)

    assert context.step.plan.query_positions.tolist() == [5, -1, 9, -1]
    target = context.main_write_target("layer.0")
    assert target.reserved_slot_mapping.tolist() == [8, 20]
    context.submit_newest_write("layer.0")

    def existing_sfa(resolution):
        assert resolution.local_sparse_indices.shape == (2, 4)
        assert resolution.hot_block_table.shape == (2, 3)
        events.append("existing_sfa")
        return torch.tensor([42])

    result = context.run_layer_attention(
        "layer.0",
        torch.zeros((2, 1, 4), dtype=torch.int32),
        existing_sfa,
    )
    context.finish()

    assert result.tolist() == [42]
    assert events == [
        "lookup_update",
        "dsa_sparse_io:region:layer.0",
        "existing_sfa",
    ]


def test_dynamic_batch_order_does_not_change_stable_request_index():
    (
        coordinator,
        cohort_key,
        plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    context = DSASparseEagerBatchContext.begin(
        coordinator,
        cohort_key,
        plan_key,
        request_ids=["request-b", "request-a"],
        request_indices=[1, 0],
        query_positions=torch.tensor([9, 5], dtype=torch.int32),
        query_counts=[1, 1],
        seq_lens=torch.tensor([10, 7], dtype=torch.int32),
        block_table=torch.tensor(
            [
                [8, 9, 10, 11, 12, 13, 14, 15],
                [0, 1, 2, 3, 4, 5, 6, 7],
            ],
            dtype=torch.int32,
        ),
    )

    assert context.step.request_indices == (1, 0)
    assert context.step.plan.query_positions.tolist() == [5, -1, 9, -1]
    assert context.step.plan.seq_lens.tolist() == [7, 10]
    assert context.step.plan.block_table.tolist() == [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
    ]
    target = context.main_write_target("layer.0")
    assert target.reserved_slot_mapping.tolist() == [20, 8]
    context.abort()


def test_dynamic_eager_batch_stages_int64_runner_positions_as_int32():
    (
        coordinator,
        cohort_key,
        plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    context = begin_dynamic_batch(
        coordinator,
        cohort_key,
        plan_key,
        query_positions_dtype=torch.int64,
    )

    assert context.step.plan.query_positions.dtype == torch.int32
    assert context.step.plan.query_positions.tolist() == [5, -1, 9, -1]
    context.abort()


def test_dynamic_eager_batch_keeps_padding_out_of_lookup_and_cache_writes():
    (
        coordinator,
        cohort_key,
        plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    context = begin_dynamic_batch(
        coordinator,
        cohort_key,
        plan_key,
        num_sfa_queries=6,
    )

    target = context.main_write_target("layer.0")
    assert target.reserved_slot_mapping.tolist() == [8, 20, -1, -1, -1, -1]
    context.submit_newest_write("layer.0")

    def existing_sfa(resolution):
        assert resolution.local_sparse_indices.shape == (6, 4)
        assert resolution.local_sparse_indices[2:].eq(-1).all()
        return torch.tensor([42])

    result = context.run_layer_attention(
        "layer.0",
        torch.zeros((6, 1, 4), dtype=torch.int32),
        existing_sfa,
    )
    context.finish()

    assert result.tolist() == [42]
    assert context.step.plan.valid_topk_counts.tolist() == [4, 0, 4, 0]


def test_dynamic_batch_begin_retires_step_when_sfa_view_copy_fails(
    monkeypatch,
):
    (
        coordinator,
        cohort_key,
        plan_key,
        _events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    original_begin_step = coordinator.begin_step

    class FailingTensor:
        def __init__(self, plan, original_metadata):
            self.plan = plan
            self.original_metadata = original_metadata

        def view(self, *_shape):
            object.__setattr__(
                self.plan,
                "batch_metadata",
                self.original_metadata,
            )
            raise RuntimeError("SFA view copy failed")

    def begin_then_fail_copy(*args, **kwargs):
        step = original_begin_step(*args, **kwargs)
        original_metadata = step.plan.batch_metadata
        object.__setattr__(
            step.plan,
            "batch_metadata",
            replace(
                original_metadata,
                write_destination_hot_row_ids=FailingTensor(
                    step.plan,
                    original_metadata,
                ),
            ),
        )
        return step

    monkeypatch.setattr(
        coordinator,
        "begin_step",
        begin_then_fail_copy,
    )
    with pytest.raises(RuntimeError, match="SFA view copy failed"):
        begin_dynamic_batch(coordinator, cohort_key, plan_key)

    monkeypatch.setattr(
        coordinator,
        "begin_step",
        original_begin_step,
    )
    next_context = begin_dynamic_batch(coordinator, cohort_key, plan_key)
    next_context.abort()


def test_layer_router_keeps_two_indexcache_cohorts_independent():
    (
        coordinator,
        first_cohort_key,
        first_plan_key,
        events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0", "layer.1"),
        freeze=False,
    )
    config = coordinator.config
    second_cohort_key = DSASparseCohortKey(
        name="shared-indexer-1",
        role="target",
    )
    second_plan_key = DSASparsePlanKey(
        token_capacity=4,
        request_capacity=2,
        query_lane_capacity=2,
        role="target",
    )
    coordinator.register_cohort(
        DSASparseCohort(
            key=second_cohort_key,
            leader_layer="layer.2",
            state=DSASparseResidencyState.allocate(
                config,
                second_cohort_key,
                device="cpu",
            ),
            plans={
                second_plan_key: DSASparsePlan.allocate(
                    config,
                    second_plan_key,
                    device="cpu",
                )
            },
        )
    )
    for layer_name in ("layer.2", "layer.3"):
        coordinator.register_layer(
            DSASparseLayerBinding(
                layer_name=layer_name,
                cohort=second_cohort_key,
                hot_cache=DSASparseLayerHotCache.allocate(
                    DSASparseLayerLayout(
                        layer_name=layer_name,
                        plane_dtypes=(torch.bfloat16, torch.bfloat16),
                        plane_row_shapes=((1, 8), (1, 2)),
                    ),
                    config,
                    device="cpu",
                ),
                io_context=f"context:{layer_name}",
                io_region=f"region:{layer_name}",
                io_completion=object(),
            )
        )
    coordinator.freeze()

    first_context = begin_dynamic_batch(
        coordinator,
        first_cohort_key,
        first_plan_key,
    )
    second_context = begin_dynamic_batch(
        coordinator,
        second_cohort_key,
        second_plan_key,
    )
    router = DSASparseEagerContextRouter(
        {
            "layer.0": first_context,
            "layer.1": first_context,
            "layer.2": second_context,
            "layer.3": second_context,
        }
    )

    for layer_name in ("layer.0", "layer.1", "layer.2", "layer.3"):
        router.main_write_target(layer_name)
        router.submit_newest_write(layer_name)
        router.run_layer_attention(
            layer_name,
            torch.zeros((2, 4), dtype=torch.int32),
            lambda resolution: torch.tensor([resolution.hot_main_cache[0].shape[0]]),
        )

    router.finish()

    assert events.count("lookup_update") == 2
    assert router.context_for("layer.0") is first_context
    assert router.context_for("layer.3") is second_context


def test_layer_router_aborts_every_cohort_after_finish_failure():
    events = []

    class RecordingContext:
        num_sfa_queries = 2

        def __init__(self, name, *, fail_finish=False):
            self.name = name
            self.fail_finish = fail_finish

        def finish(self):
            events.append(f"finish:{self.name}")
            if self.fail_finish:
                raise RuntimeError("finish failed")

        def abort(self):
            events.append(f"abort:{self.name}")

    first_context = RecordingContext("first")
    second_context = RecordingContext("second", fail_finish=True)
    router = DSASparseEagerContextRouter(
        {
            "layer.0": first_context,
            "layer.1": second_context,
        }
    )

    with pytest.raises(RuntimeError, match="finish failed"):
        router.finish()

    assert events == [
        "finish:first",
        "finish:second",
        "abort:first",
        "abort:second",
    ]


def test_unified_io_failure_poisons_coordinator():
    (
        coordinator,
        cohort_key,
        plan_key,
        events,
        transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    coordinator.io_operator = FailingIOOperator(
        events,
        transfer_counts,
    )
    context = begin_dynamic_batch(coordinator, cohort_key, plan_key)
    context.submit_newest_write("layer.0")

    with pytest.raises(RuntimeError, match="unified I/O completion failed"):
        context.run_layer_attention(
            "layer.0",
            torch.zeros((2, 4), dtype=torch.int32),
            lambda _resolution: torch.tensor([42]),
        )
    context.abort()
    with pytest.raises(RuntimeError, match="poisoned"):
        begin_dynamic_batch(coordinator, cohort_key, plan_key)


def test_lookup_update_failure_poisons_coordinator():
    (
        coordinator,
        cohort_key,
        plan_key,
        events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    coordinator.index_operator = FailingIndexOperator(
        events,
        has_misses=False,
    )
    context = begin_dynamic_batch(coordinator, cohort_key, plan_key)
    context.submit_newest_write("layer.0")

    with pytest.raises(RuntimeError, match="lookup/update failed"):
        context.run_layer_attention(
            "layer.0",
            torch.zeros((2, 4), dtype=torch.int32),
            lambda _resolution: torch.tensor([42]),
        )
    context.abort()
    with pytest.raises(RuntimeError, match="poisoned"):
        begin_dynamic_batch(coordinator, cohort_key, plan_key)


def test_first_unified_io_failure_remains_poisoned_after_abort():
    (
        coordinator,
        cohort_key,
        plan_key,
        events,
        transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    coordinator.io_operator = FailingOnceIOOperator(
        events,
        transfer_counts,
    )
    context = begin_dynamic_batch(coordinator, cohort_key, plan_key)
    context.submit_newest_write("layer.0")

    with pytest.raises(RuntimeError, match="completion failed once"):
        context.run_layer_attention(
            "layer.0",
            torch.zeros((2, 4), dtype=torch.int32),
            lambda _resolution: torch.tensor([42]),
        )
    context.abort()

    with pytest.raises(RuntimeError, match="poisoned"):
        begin_dynamic_batch(coordinator, cohort_key, plan_key)


def test_failed_attention_can_abort_after_unified_io_returns():
    (
        coordinator,
        cohort_key,
        plan_key,
        events,
        _transfer_counts,
    ) = build_coordinator(
        has_misses=False,
        layer_names=("layer.0",),
    )
    context = begin_dynamic_batch(coordinator, cohort_key, plan_key)
    context.submit_newest_write("layer.0")

    def failed_sfa(_resolution):
        raise RuntimeError("SFA failed")

    with pytest.raises(RuntimeError, match="SFA failed"):
        context.run_layer_attention(
            "layer.0",
            torch.zeros((2, 4), dtype=torch.int32),
            failed_sfa,
        )

    assert events[-1] == "dsa_sparse_io:region:layer.0"
    context.abort()

    next_context = begin_dynamic_batch(coordinator, cohort_key, plan_key)
    next_context.abort()
