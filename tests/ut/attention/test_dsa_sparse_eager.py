# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass, field

import pytest
import torch

from vllm_ascend.attention.dsa_sparse import (
    DSASparseCacheConfig,
    DSASparseCohort,
    DSASparseCohortKey,
    DSASparseEagerBatchContext,
    DSASparseEagerCoordinator,
    DSASparseLayerBinding,
    DSASparseLayerHotCache,
    DSASparseLookupBatch,
    DSASparseLookupOutput,
    DSASparseLookupState,
)
from vllm_ascend.dsa_sparse_constants import DSA_SPARSE_QUERY_WIDTH


@dataclass
class RecordingLookupOperator:
    calls: list[tuple[object, DSASparseLookupBatch]] = field(
        default_factory=list
    )
    failure: Exception | None = None

    def lookup(self, *, state, batch):
        self.calls.append((state, batch))
        if self.failure is not None:
            raise self.failure
        slot_out = torch.full_like(batch.query_index, 77)
        miss_out = batch.lookup_mask.clone()
        return DSASparseLookupOutput(
            slot_out=slot_out,
            miss_out=miss_out,
        )


@dataclass
class RecordingIOOperator:
    calls: list[dict[str, object]] = field(default_factory=list)
    failure: Exception | None = None

    def dsa_sparse_io(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure


def build_coordinator(
    *,
    lookup_operator: RecordingLookupOperator | None = None,
    io_operator: RecordingIOOperator | None = None,
    layer_names: tuple[str, ...] = ("layer.0", "layer.1"),
    freeze: bool = True,
):
    config = DSASparseCacheConfig(
        max_num_seqs=3,
        max_model_len=512,
        block_size=128,
        index_topk=DSA_SPARSE_QUERY_WIDTH,
    )
    lookup_operator = lookup_operator or RecordingLookupOperator()
    io_operator = io_operator or RecordingIOOperator()
    coordinator = DSASparseEagerCoordinator(
        config,
        lookup_operator=lookup_operator,
        io_operator=io_operator,
    )
    cohort_key = DSASparseCohortKey("index-cache-0", "target")
    coordinator.register_cohort(
        DSASparseCohort(
            key=cohort_key,
            leader_layer=layer_names[0],
            state=DSASparseLookupState.allocate(
                config,
                cohort_key,
                device="cpu",
            ),
        )
    )
    for layer_name in layer_names:
        coordinator.register_layer(
            DSASparseLayerBinding(
                layer_name=layer_name,
                cohort=cohort_key,
                hot_cache=DSASparseLayerHotCache(
                    layer_name=layer_name,
                    planes=(torch.empty((1, 1)),),
                ),
                io_context=f"context:{layer_name}",
                io_region=f"region:{layer_name}",
                io_completion=object(),
            )
        )
    if freeze:
        coordinator.freeze()
    return coordinator, cohort_key, lookup_operator, io_operator


def build_metadata(coordinator):
    return coordinator.build_step_metadata(
        request_ids=["request-a", "request-c"],
        query_positions=torch.tensor([130, 260], dtype=torch.int64),
        seq_lens=torch.tensor([131, 261], dtype=torch.int32),
        block_table=torch.tensor(
            [
                [10, 11, 12, 13],
                [20, 21, 22, 23],
            ],
            dtype=torch.int32,
        ),
    )


def admit_non_contiguous_requests(coordinator):
    coordinator.acquire_request("request-a")
    coordinator.acquire_request("request-b")
    coordinator.acquire_request("request-c")
    coordinator.release_request("request-b")


def semantic_topk():
    topk = torch.full(
        (2, DSA_SPARSE_QUERY_WIDTH),
        -1,
        dtype=torch.int32,
    )
    topk[0, :2] = torch.tensor([10, 130])
    topk[1, :2] = torch.tensor([20, 258])
    return topk


def test_compact_lookup_masks_tail_and_padding_and_fans_out_per_layer():
    coordinator, cohort_key, lookup, io = build_coordinator()
    admit_non_contiguous_requests(coordinator)
    metadata = build_metadata(coordinator)
    context = DSASparseEagerBatchContext.begin(
        coordinator,
        cohort_key,
        metadata=metadata,
    )

    assert metadata.req_pool_entries.tolist() == [0, 2]
    assert metadata.req_pool_entries.dtype == torch.int32
    assert metadata.req_pool_entries.is_contiguous()
    assert metadata.dense_tail_starts.tolist() == [128, 256]
    assert metadata.resident_tail_starts.tolist() == [10240, 10240]
    assert metadata.write_destination_slots.tolist() == [
        10242,
        2 * 10368 + 10244,
    ]

    resolutions = []
    for layer_name in ("layer.0", "layer.1"):
        target = context.main_write_target(layer_name)
        assert target.slot_mapping is metadata.write_destination_slots
        context.submit_newest_write(layer_name)
        context.run_layer_attention(
            layer_name,
            semantic_topk(),
            lambda resolution: resolutions.append(resolution)
            or torch.empty(0),
        )
    context.finish()

    assert len(lookup.calls) == 1
    state, batch = lookup.calls[0]
    assert state is coordinator.get_cohort(cohort_key).state
    assert batch.req_pool_entries is metadata.req_pool_entries
    assert batch.query_index.shape == (2, DSA_SPARSE_QUERY_WIDTH)
    assert batch.query_index.dtype == torch.int32
    assert batch.query_index.is_contiguous()
    assert batch.lookup_mask[0, :3].tolist() == [1, 0, 0]
    assert batch.lookup_mask[1, :3].tolist() == [1, 0, 0]

    assert len(io.calls) == 2
    assert io.calls[0]["slot_out"] is io.calls[1]["slot_out"]
    assert io.calls[0]["miss_out"] is io.calls[1]["miss_out"]
    assert io.calls[0]["req_pool_entries"] is metadata.req_pool_entries
    assert io.calls[1]["req_pool_entries"] is metadata.req_pool_entries
    assert resolutions[0].attention_indices[0, :3].tolist() == [
        77,
        10242,
        -1,
    ]
    assert resolutions[0].attention_indices[1, :3].tolist() == [
        77,
        10242,
        -1,
    ]
    assert resolutions[0].attention_indices is resolutions[1].attention_indices
    assert resolutions[0].hot_main_cache is not resolutions[1].hot_main_cache


def test_follower_does_not_invoke_lookup_before_leader():
    coordinator, cohort_key, lookup, _ = build_coordinator()
    admit_non_contiguous_requests(coordinator)
    context = DSASparseEagerBatchContext.begin(
        coordinator,
        cohort_key,
        metadata=build_metadata(coordinator),
    )
    context.submit_newest_write("layer.1")

    with pytest.raises(RuntimeError, match="leader"):
        context.run_layer_attention(
            "layer.1",
            semantic_topk(),
            lambda resolution: torch.empty(0),
        )
    assert lookup.calls == []
    context.abort()


def test_request_release_resets_all_lookup_state_rows():
    coordinator, cohort_key, _, _ = build_coordinator()
    request_index = coordinator.acquire_request("request")
    state = coordinator.get_cohort(cohort_key).state
    state.index[request_index, 5] = 5
    state.slot_to_index[request_index, 6] = 6
    state.free_slots[request_index].fill_(-1)
    state.free_head[request_index].fill_(9)

    coordinator.release_request("request")

    assert state.index[request_index].eq(-1).all()
    assert state.slot_to_index[request_index].eq(-1).all()
    assert state.free_slots[request_index, 0].item() == 8192
    assert state.free_slots[request_index, -1].item() == 10239
    assert state.free_head[request_index].eq(0).all()


def test_lookup_failure_poisons_coordinator():
    lookup = RecordingLookupOperator(
        failure=RuntimeError("lookup failed")
    )
    coordinator, cohort_key, _, _ = build_coordinator(
        lookup_operator=lookup
    )
    admit_non_contiguous_requests(coordinator)
    context = DSASparseEagerBatchContext.begin(
        coordinator,
        cohort_key,
        metadata=build_metadata(coordinator),
    )
    context.submit_newest_write("layer.0")

    with pytest.raises(RuntimeError, match="lookup failed"):
        context.run_layer_attention(
            "layer.0",
            semantic_topk(),
            lambda resolution: torch.empty(0),
        )
    context.abort()
    with pytest.raises(RuntimeError, match="poisoned"):
        coordinator.acquire_request("request-d")


def test_io_failure_prevents_normal_step_finish():
    io = RecordingIOOperator(
        failure=RuntimeError("io failed")
    )
    coordinator, cohort_key, _, _ = build_coordinator(
        io_operator=io
    )
    admit_non_contiguous_requests(coordinator)
    context = DSASparseEagerBatchContext.begin(
        coordinator,
        cohort_key,
        metadata=build_metadata(coordinator),
    )
    context.submit_newest_write("layer.0")

    with pytest.raises(RuntimeError, match="io failed"):
        context.run_layer_attention(
            "layer.0",
            semantic_topk(),
            lambda resolution: torch.empty(0),
        )
    with pytest.raises(RuntimeError, match="every layer"):
        context.finish()
    context.abort()
