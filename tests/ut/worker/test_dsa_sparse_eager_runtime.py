# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from unittest.mock import patch

import pytest
import torch

from tests.ut.attention.test_dsa_sparse_eager import (
    RecordingIndexOperator,
    build_coordinator,
)
from vllm_ascend import dsa_sparse_probe
from vllm_ascend.attention.dsa_sparse import (
    DSASparseCacheConfig,
    DSASparseCohort,
    DSASparseCohortKey,
    DSASparseLayerBinding,
    DSASparseLayerHotCache,
    DSASparseLayerLayout,
    DSASparsePlan,
    DSASparsePlanKey,
    DSASparseResidencyState,
)
from vllm_ascend.worker.dsa_sparse_eager import (
    DSASparseEagerCohortDescriptor,
    DSASparseEagerCohortLayout,
    DSASparseEagerRuntime,
    create_dsa_sparse_eager_mock_runtime,
)
from vllm_ascend.worker.dsa_sparse_memory import (
    calculate_dsa_sparse_fixed_hbm_bytes,
)


@dataclass
class FakeMetadata:
    num_input_tokens: int = 2
    num_actual_tokens: int = 2
    seq_lens: torch.Tensor = None
    block_table: torch.Tensor = None
    dsa_sparse_context: object | None = None

    def __post_init__(self):
        if self.seq_lens is None:
            self.seq_lens = torch.tensor([7, 10], dtype=torch.int32)
        if self.block_table is None:
            self.block_table = torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [8, 9, 10, 11, 12, 13, 14, 15],
                ],
                dtype=torch.int32,
            )


class TrackingMetadata(FakeMetadata):
    def __init__(self):
        self.context_assignments = []
        self._dsa_sparse_context = None
        super().__init__()
        self.context_assignments.clear()

    @property
    def dsa_sparse_context(self):
        return self._dsa_sparse_context

    @dsa_sparse_context.setter
    def dsa_sparse_context(self, value):
        self.context_assignments.append(value)
        self._dsa_sparse_context = value


class FailingAttachMetadata(FakeMetadata):
    def __init__(self):
        self._dsa_sparse_context = None
        super().__init__()

    @property
    def dsa_sparse_context(self):
        return self._dsa_sparse_context

    @dsa_sparse_context.setter
    def dsa_sparse_context(self, value):
        if value is not None:
            raise RuntimeError("metadata attach failed")
        self._dsa_sparse_context = value


def _descriptor(cohort_key, plan_key, *layer_names):
    return DSASparseEagerCohortDescriptor(
        cohort_key=cohort_key,
        plan_key=plan_key,
        layer_names=layer_names,
        leader_layer=layer_names[0],
    )


def _batch_arguments(layer_metadata):
    return {
        "request_ids": ["request-a", "request-b"],
        "query_positions": torch.tensor([5, 9], dtype=torch.int32),
        "query_counts": [1, 1],
        "layer_metadata": layer_metadata,
    }


def _run_all_layers(router, *layer_names):
    for layer_name in layer_names:
        router.main_write_target(layer_name)
        router.submit_newest_write(layer_name)
        router.run_layer_attention(
            layer_name,
            torch.zeros((2, 4), dtype=torch.int32),
            lambda resolution: torch.tensor([resolution.hot_main_cache[0].shape[0]]),
        )


def _add_second_cohort(coordinator):
    config = coordinator.config
    first_cohort = coordinator.get_cohort(
        DSASparseCohortKey(
            name="shared-indexer-0",
            role="target",
        )
    )
    shared_batch_metadata = next(
        iter(first_cohort.plans.values())
    ).batch_metadata
    cohort_key = DSASparseCohortKey(
        name="shared-indexer-1",
        role="target",
    )
    plan_key = DSASparsePlanKey(
        token_capacity=4,
        request_capacity=2,
        query_lane_capacity=2,
        role="target",
    )
    coordinator.register_cohort(
        DSASparseCohort(
            key=cohort_key,
            leader_layer="layer.2",
            state=DSASparseResidencyState.allocate(
                config,
                cohort_key,
                device="cpu",
            ),
            plans={
                plan_key: DSASparsePlan.allocate(
                    config,
                    plan_key,
                    device="cpu",
                    batch_metadata=shared_batch_metadata,
                )
            },
        )
    )
    for layer_name in ("layer.2", "layer.3"):
        coordinator.register_layer(
            DSASparseLayerBinding(
                layer_name=layer_name,
                cohort=cohort_key,
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
    return cohort_key, plan_key


def _logical_tensor_bytes(*values) -> int:
    seen_tensors: set[int] = set()

    def visit(value) -> int:
        if isinstance(value, torch.Tensor):
            identity = id(value)
            if identity in seen_tensors:
                return 0
            seen_tensors.add(identity)
            return value.numel() * value.element_size()
        if is_dataclass(value):
            return sum(visit(getattr(value, field.name)) for field in fields(value))
        if isinstance(value, Mapping):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        return 0

    return sum(visit(value) for value in values)


def test_mock_factory_allocates_resources_and_runs_injected_lookup():
    config = DSASparseCacheConfig(
        max_num_seqs=2,
        max_model_len=32,
        block_size=4,
        device_buffer_size=4,
        max_query_tokens_per_request=1,
        index_topk=4,
    )
    layer_layouts = (
        DSASparseLayerLayout(
            layer_name="layer.0",
            plane_dtypes=(torch.bfloat16, torch.bfloat16),
            plane_row_shapes=((1, 8), (1, 2)),
        ),
        DSASparseLayerLayout(
            layer_name="layer.1",
            plane_dtypes=(torch.bfloat16, torch.bfloat16),
            plane_row_shapes=((1, 8), (1, 2)),
        ),
    )
    lookup_events = []
    with (
        patch.object(
            dsa_sparse_probe,
            "is_enabled",
            return_value=True,
        ),
        patch.object(
            dsa_sparse_probe,
            "emit",
        ) as mock_probe_emit,
    ):
        runtime = create_dsa_sparse_eager_mock_runtime(
            config,
            [
                DSASparseEagerCohortLayout(
                    cohort_name="target-indexer-0",
                    layer_layouts=layer_layouts,
                )
            ],
            device="cpu",
            index_operator=RecordingIndexOperator(
                lookup_events,
                has_misses=False,
            ),
        )
    assert runtime.uses_mock_lifecycle
    probe_events = [
        call.args[0]
        for call in mock_probe_emit.call_args_list
    ]
    assert probe_events == [
        "hot_cache_registered",
        "hot_cache_registered",
        "runtime_ready",
    ]
    runtime_ready = mock_probe_emit.call_args_list[-1]
    assert runtime_ready.kwargs == {
        "cohort_count": 1,
        "layer_count": 2,
        "index_topk": 4,
        "cohorts": [
            {
                "name": "target-indexer-0",
                "layers": ["layer.0", "layer.1"],
            }
        ],
    }

    descriptor = runtime.cohort_descriptors[0]
    cohort = runtime.coordinator.get_cohort(descriptor.cohort_key)
    assert descriptor.layer_names == ("layer.0", "layer.1")
    assert cohort.leader_layer == "layer.0"
    assert cohort.state.token_to_hot.shape == (2, 32)
    assert cohort.plans[descriptor.plan_key].key.request_capacity == 2
    layer_0 = runtime.coordinator.get_layer_binding(
        descriptor.cohort_key,
        "layer.0",
    )
    layer_1 = runtime.coordinator.get_layer_binding(
        descriptor.cohort_key,
        "layer.1",
    )
    assert layer_0.hot_cache.planes[0].data_ptr() != (layer_1.hot_cache.planes[0].data_ptr())

    fixed_objects = [
        cohort.state,
        *cohort.plans.values(),
        layer_0.hot_cache,
        layer_1.hot_cache,
    ]
    breakdown = calculate_dsa_sparse_fixed_hbm_bytes(
        config,
        layer_layouts,
        cohort_count=1,
        max_sfa_queries=2,
    )
    assert _logical_tensor_bytes(*fixed_objects) == (breakdown.core_fixed_tensor_bytes)

    metadata = FakeMetadata(
        num_input_tokens=1,
        num_actual_tokens=1,
    )
    batch_arguments = {
        "request_ids": ["request-a"],
        "query_positions": torch.tensor([5], dtype=torch.int32),
        "query_counts": [1],
        "layer_metadata": {
            "layer.0": metadata,
            "layer.1": metadata,
        },
    }
    with pytest.raises(KeyError, match="does not own"):
        runtime.begin_target_batch(**batch_arguments)

    request_index = runtime.admit_mock_request("request-a")
    assert runtime.has_mock_request("request-a")
    assert request_index == 0
    with runtime.begin_target_batch(**batch_arguments) as router:
        for layer_name in ("layer.0", "layer.1"):
            router.main_write_target(layer_name)
            router.submit_newest_write(layer_name)
            router.run_layer_attention(
                layer_name,
                torch.zeros((1, 4), dtype=torch.int32),
                lambda resolution: torch.tensor(
                    [resolution.hot_main_cache[0].shape[0]]
                ),
            )
    assert lookup_events == ["lookup_update"]
    runtime.retire_mock_request("request-a", preempted=False)
    assert not runtime.has_mock_request("request-a")


def test_mock_admission_failure_rolls_back_generation_state():
    config = DSASparseCacheConfig(
        max_num_seqs=1,
        max_model_len=16,
        block_size=4,
        device_buffer_size=4,
        max_query_tokens_per_request=1,
        index_topk=4,
    )
    runtime = create_dsa_sparse_eager_mock_runtime(
        config,
        [
            DSASparseEagerCohortLayout(
                cohort_name="target-indexer-0",
                layer_layouts=(
                    DSASparseLayerLayout(
                        layer_name="layer.0",
                        plane_dtypes=(torch.bfloat16,),
                        plane_row_shapes=((1, 8),),
                    ),
                ),
            ),
        ],
        device="cpu",
        index_operator=RecordingIndexOperator([], has_misses=False),
    )
    first_request_index = runtime.admit_mock_request("request-a")

    with pytest.raises(RuntimeError, match="No free"):
        runtime.admit_mock_request("request-b")

    runtime.retire_mock_request("request-a", preempted=False)
    second_request_index = runtime.admit_mock_request("request-b")

    assert first_request_index == second_request_index == 0
    runtime.retire_mock_request("request-b", preempted=False)


def test_success_attaches_one_router_to_shared_metadata_and_finishes():
    coordinator, cohort_key, plan_key, events, _ = build_coordinator(
        has_misses=False,
    )
    runtime = DSASparseEagerRuntime(
        coordinator,
        [_descriptor(cohort_key, plan_key, "layer.0", "layer.1")],
    )
    assert not runtime.uses_mock_lifecycle
    metadata = TrackingMetadata()

    with runtime.begin_target_batch(
        **_batch_arguments(
            {
                "layer.0": metadata,
                "layer.1": metadata,
            }
        )
    ) as router:
        assert metadata.dsa_sparse_context is router
        assert router.context_for("layer.0") is router.context_for("layer.1")
        _run_all_layers(router, "layer.0", "layer.1")

    assert metadata.dsa_sparse_context is None
    assert len(metadata.context_assignments) == 2
    assert events.count("lookup_update") == 1


def test_multiple_cohorts_share_router_but_keep_contexts_independent():
    coordinator, first_key, first_plan, events, _ = build_coordinator(
        has_misses=False,
        freeze=False,
    )
    second_key, second_plan = _add_second_cohort(coordinator)
    coordinator.freeze()
    runtime = DSASparseEagerRuntime(
        coordinator,
        [
            _descriptor(first_key, first_plan, "layer.0", "layer.1"),
            _descriptor(second_key, second_plan, "layer.2", "layer.3"),
        ],
    )
    first_metadata = FakeMetadata()
    second_metadata = FakeMetadata()

    with runtime.begin_target_batch(
        **_batch_arguments(
            {
                "layer.0": first_metadata,
                "layer.1": first_metadata,
                "layer.2": second_metadata,
                "layer.3": second_metadata,
            }
        )
    ) as router:
        assert first_metadata.dsa_sparse_context is router
        assert second_metadata.dsa_sparse_context is router
        assert router.context_for("layer.0") is not router.context_for("layer.2")
        _run_all_layers(
            router,
            "layer.0",
            "layer.1",
            "layer.2",
            "layer.3",
        )

    assert first_metadata.dsa_sparse_context is None
    assert second_metadata.dsa_sparse_context is None
    assert events.count("lookup_update") == 2


def test_forward_exception_aborts_all_contexts_and_detaches_metadata():
    coordinator, cohort_key, plan_key, _events, _ = build_coordinator(
        has_misses=False,
    )
    runtime = DSASparseEagerRuntime(
        coordinator,
        [_descriptor(cohort_key, plan_key, "layer.0", "layer.1")],
    )
    metadata = FakeMetadata()
    arguments = _batch_arguments(
        {
            "layer.0": metadata,
            "layer.1": metadata,
        }
    )

    with (
        pytest.raises(RuntimeError, match="model forward failed"),
        runtime.begin_target_batch(**arguments) as router,
    ):
        router.main_write_target("layer.0")
        router.submit_newest_write("layer.0")
        raise RuntimeError("model forward failed")

    assert metadata.dsa_sparse_context is None
    next_execution = runtime.begin_target_batch(**arguments)
    with pytest.raises(RuntimeError, match="pending layers"), next_execution:
        pass


def test_finish_failure_aborts_remaining_contexts_and_detaches_metadata():
    coordinator, cohort_key, plan_key, _events, _ = build_coordinator(
        has_misses=False,
    )
    runtime = DSASparseEagerRuntime(
        coordinator,
        [_descriptor(cohort_key, plan_key, "layer.0", "layer.1")],
    )
    metadata = FakeMetadata()
    arguments = _batch_arguments(
        {
            "layer.0": metadata,
            "layer.1": metadata,
        }
    )

    with (
        pytest.raises(RuntimeError, match="pending layers"),
        runtime.begin_target_batch(**arguments),
    ):
        pass

    assert metadata.dsa_sparse_context is None
    with runtime.begin_target_batch(**arguments) as router:
        _run_all_layers(router, "layer.0", "layer.1")


def test_partial_begin_failure_aborts_previously_started_cohorts():
    coordinator, first_key, first_plan, _events, _ = build_coordinator(
        has_misses=False,
        freeze=False,
    )
    second_key, second_plan = _add_second_cohort(coordinator)
    coordinator.freeze()
    runtime = DSASparseEagerRuntime(
        coordinator,
        [
            _descriptor(first_key, first_plan, "layer.0", "layer.1"),
            _descriptor(second_key, second_plan, "layer.2", "layer.3"),
        ],
    )
    first_metadata = FakeMetadata()
    invalid_metadata = FakeMetadata(num_actual_tokens=1)
    layer_metadata = {
        "layer.0": first_metadata,
        "layer.1": first_metadata,
        "layer.2": invalid_metadata,
        "layer.3": invalid_metadata,
    }

    with pytest.raises(ValueError, match="num_actual_tokens"):
        runtime.begin_target_batch(**_batch_arguments(layer_metadata))

    assert first_metadata.dsa_sparse_context is None
    valid_metadata = FakeMetadata()
    layer_metadata["layer.2"] = valid_metadata
    layer_metadata["layer.3"] = valid_metadata
    with runtime.begin_target_batch(**_batch_arguments(layer_metadata)) as router:
        _run_all_layers(
            router,
            "layer.0",
            "layer.1",
            "layer.2",
            "layer.3",
        )


def test_partial_attach_failure_detaches_and_aborts_all_contexts():
    coordinator, cohort_key, plan_key, _events, _ = build_coordinator(
        has_misses=False,
    )
    runtime = DSASparseEagerRuntime(
        coordinator,
        [_descriptor(cohort_key, plan_key, "layer.0", "layer.1")],
    )
    attached_metadata = FakeMetadata()
    failing_metadata = FailingAttachMetadata()

    with pytest.raises(RuntimeError, match="metadata attach failed"):
        runtime.begin_target_batch(
            **_batch_arguments(
                {
                    "layer.0": attached_metadata,
                    "layer.1": failing_metadata,
                }
            )
        )

    assert attached_metadata.dsa_sparse_context is None
    valid_metadata = FakeMetadata()
    with runtime.begin_target_batch(
        **_batch_arguments(
            {
                "layer.0": attached_metadata,
                "layer.1": valid_metadata,
            }
        )
    ) as router:
        _run_all_layers(router, "layer.0", "layer.1")


def test_existing_metadata_context_is_rejected_without_beginning():
    coordinator, cohort_key, plan_key, _events, _ = build_coordinator(
        has_misses=False,
    )
    runtime = DSASparseEagerRuntime(
        coordinator,
        [_descriptor(cohort_key, plan_key, "layer.0", "layer.1")],
    )
    existing_context = object()
    metadata = FakeMetadata(dsa_sparse_context=existing_context)

    with pytest.raises(RuntimeError, match="already owns"):
        runtime.begin_target_batch(
            **_batch_arguments(
                {
                    "layer.0": metadata,
                    "layer.1": metadata,
                }
            )
        )

    assert metadata.dsa_sparse_context is existing_context
    metadata.dsa_sparse_context = None
    with runtime.begin_target_batch(
        **_batch_arguments(
            {
                "layer.0": metadata,
                "layer.1": metadata,
            }
        )
    ) as router:
        _run_all_layers(router, "layer.0", "layer.1")
