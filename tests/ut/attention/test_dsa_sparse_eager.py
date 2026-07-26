# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import FrozenInstanceError, dataclass

import pytest
import torch

from vllm_ascend.attention.dsa_sparse import (
    CacheSeatManager,
    DSASparseCacheConfig,
    DSASparseCohort,
    DSASparseCohortKey,
    DSASparseEagerCoordinator,
    DSASparseLayerBinding,
    DSASparseLayerHotCache,
    DSASparseLayerLayout,
    DSASparsePlan,
    DSASparsePlanKey,
    DSASparseResidencyState,
    DSASparseResolution,
)


@dataclass
class RecordingIndexOperator:
    events: list[str]
    has_misses: bool

    def prepare_newest(self, *, state, plan):
        self.events.append("prepare_newest")
        config_evictable_slots = state.lru_slots.shape[1]
        hot_stride = plan.hot_block_table.shape[1] * 4
        row_mapping = plan.row_mapping

        plan.hot_block_table.fill_(-1)
        plan.newest_destination_hot_row_ids.fill_(-1)
        plan.write_global_slots.fill_(-1)
        plan.write_destination_hot_row_ids.fill_(-1)
        plan.write_valid_mask.zero_()
        for query_index in range(plan.key.token_capacity):
            row = int(plan.query_to_row[query_index])
            lane = int(plan.query_to_lane[query_index])
            if not bool(plan.query_valid_mask[query_index]):
                continue
            seat = int(row_mapping.row_to_cache_seat[row])
            local_slot = config_evictable_slots + lane
            destination_row = seat * hot_stride + local_slot
            plan.newest_destination_hot_row_ids[query_index] = destination_row
            plan.write_destination_hot_row_ids[row, lane] = destination_row
            token_position = int(plan.query_positions[query_index])
            logical_block = token_position // 4
            token_offset = token_position % 4
            physical_block = int(plan.block_table[row, logical_block])
            plan.write_global_slots[row, lane] = physical_block * 4 + token_offset
            plan.write_valid_mask[row, lane] = True

        for row in range(plan.key.request_capacity):
            if not bool(row_mapping.row_active[row]):
                continue
            seat = int(row_mapping.row_to_cache_seat[row])
            plan.hot_block_table[row].copy_(
                torch.arange(
                    seat * plan.hot_block_table.shape[1],
                    (seat + 1) * plan.hot_block_table.shape[1],
                    dtype=plan.hot_block_table.dtype,
                )
            )

    def lookup(self, *, state, plan):
        del state
        self.events.append("lookup")
        plan.resolved_hot_indices.zero_()
        plan.read_source_global_slots.fill_(-1)
        plan.read_local_hot_slot_ids.fill_(-1)
        plan.read_destination_hot_row_ids.fill_(-1)
        plan.read_valid_mask.zero_()
        if self.has_misses:
            plan.read_source_global_slots[0, 0, 0] = 7
            plan.read_local_hot_slot_ids[0, 0, 0] = 0
            plan.read_destination_hot_row_ids[0, 0, 0] = 0
            plan.read_valid_mask[0, 0, 0] = True


@dataclass
class RecordingIOOperator:
    events: list[str]
    transfer_counts: list[int]

    def read_async(
        self,
        *,
        context,
        region,
        source_global_slots,
        destination_hot_row_ids,
        valid_mask,
        hot_planes,
        completion,
    ):
        del (
            context,
            source_global_slots,
            destination_hot_row_ids,
            hot_planes,
            completion,
        )
        self.events.append(f"read_async:{region}")
        self.transfer_counts.append(int(valid_mask.sum()))

    def wait_read(self, *, context, completion, hot_planes):
        del context, completion, hot_planes
        self.events.append("wait_read")

    def write_async(
        self,
        *,
        context,
        region,
        destination_global_slots,
        source_hot_row_ids,
        valid_mask,
        hot_planes,
        completion,
    ):
        del (
            context,
            destination_global_slots,
            source_hot_row_ids,
            valid_mask,
            hot_planes,
            completion,
        )
        self.events.append(f"write_async:{region}")

    def wait_write(self, *, context, completion, hot_planes):
        del context, completion, hot_planes
        self.events.append("wait_write")


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
        seat_manager=CacheSeatManager(config.max_num_seqs),
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
                read_completion=object(),
                write_completion=object(),
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
        "prepare_newest",
        "write_async:region:layer.0",
        "lookup",
        "read_async:region:layer.0",
        "wait_read",
        "existing_sfa",
        "wait_write",
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

    assert events.count("lookup") == 1
    assert "read_async:region:layer.0" in events
    assert "read_async:region:layer.1" in events
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
                read_completion=object(),
                write_completion=object(),
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
            read_completion=object(),
            write_completion=object(),
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
