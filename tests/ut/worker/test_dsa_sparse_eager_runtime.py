# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass

import pytest
import torch

from tests.ut.attention.test_dsa_sparse_eager import (
    RecordingLookupOperator,
)
from vllm_ascend.attention.dsa_sparse import (
    DSASparseCacheConfig,
    DSASparseLayerLayout,
)
from vllm_ascend.attention.dsa_sparse_io import (
    MockDSASparseIOOperator,
)
from vllm_ascend.attention.dsa_sparse_pd import (
    DSASparsePDHandoff,
    build_dsa_sparse_resident_token_ids,
)
from vllm_ascend.dsa_sparse_constants import DSA_SPARSE_QUERY_WIDTH
from vllm_ascend.ops.dsa_sparse import TorchDSASparseLookupOperator
from vllm_ascend.worker.dsa_sparse_eager import (
    DSASparseEagerCohortLayout,
    begin_dsa_sparse_producer_execution,
    create_dsa_sparse_eager_mock_runtime,
)


@dataclass
class FakeMetadata:
    num_input_tokens: int = 2
    num_actual_tokens: int = 2
    seq_lens: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    dsa_sparse_context: object | None = None
    dsa_sparse_producer_context: object | None = None

    def __post_init__(self):
        if self.seq_lens is None:
            self.seq_lens = torch.tensor(
                [131, 261],
                dtype=torch.int32,
            )
        if self.block_table is None:
            self.block_table = torch.tensor(
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                ],
                dtype=torch.int32,
            )


def config():
    return DSASparseCacheConfig(
        max_num_seqs=3,
        max_model_len=512,
        block_size=128,
        index_topk=DSA_SPARSE_QUERY_WIDTH,
    )


def layout(name, layer_name):
    return DSASparseEagerCohortLayout(
        cohort_name=name,
        layer_layouts=(
            DSASparseLayerLayout(
                layer_name=layer_name,
                plane_dtypes=(torch.bfloat16,),
                plane_row_shapes=((1,),),
            ),
        ),
    )


def topk():
    result = torch.full(
        (2, DSA_SPARSE_QUERY_WIDTH),
        -1,
        dtype=torch.int32,
    )
    result[:, 0] = 10
    return result


class RecordingIOOperator(MockDSASparseIOOperator):
    def __init__(self):
        self.publications = []
        self.hot_initializations = []

    def publish_main(self, **kwargs):
        super().publish_main(**kwargs)
        self.publications.append(kwargs)

    def initialize_hot_cache(self, **kwargs):
        super().initialize_hot_cache(**kwargs)
        self.hot_initializations.append(kwargs)


def test_runtime_shares_compact_step_metadata_and_calls_each_cohort_once():
    lookup = RecordingLookupOperator()
    runtime = create_dsa_sparse_eager_mock_runtime(
        config(),
        (
            layout("cohort-0", "layer.0"),
            layout("cohort-1", "layer.1"),
        ),
        device="cpu",
        lookup_operator=lookup,
    )
    runtime.admit_mock_request("request-a")
    runtime.admit_mock_request("request-b")
    metadata_0 = FakeMetadata()
    metadata_1 = FakeMetadata()

    with runtime.begin_target_batch(
        request_ids=("request-a", "request-b"),
        query_positions=torch.tensor([130, 260]),
        layer_metadata={
            "layer.0": metadata_0,
            "layer.1": metadata_1,
        },
    ) as router:
        contexts = router.contexts
        assert len(contexts) == 2
        assert contexts[0].step.metadata is contexts[1].step.metadata
        for layer_name in ("layer.0", "layer.1"):
            router.main_write_target(layer_name)
            router.submit_newest_write(layer_name)
            router.run_layer_attention(
                layer_name,
                topk(),
                lambda resolution: torch.empty(0),
            )

    assert len(lookup.calls) == 2
    assert metadata_0.dsa_sparse_context is None
    assert metadata_1.dsa_sparse_context is None
    runtime.retire_mock_request("request-a", preempted=False)
    runtime.retire_mock_request("request-b", preempted=False)


def test_runtime_requires_one_query_position_per_request():
    runtime = create_dsa_sparse_eager_mock_runtime(
        config(),
        (layout("cohort-0", "layer.0"),),
        device="cpu",
        lookup_operator=RecordingLookupOperator(),
    )
    runtime.admit_mock_request("request-a")
    runtime.admit_mock_request("request-b")

    with pytest.raises(ValueError, match="one token per request"):
        runtime.begin_target_batch(
            request_ids=("request-a", "request-b"),
            query_positions=torch.tensor([130, 131, 260]),
            layer_metadata={"layer.0": FakeMetadata()},
        )


def test_default_runtime_uses_fused_torch_lookup_operator():
    runtime = create_dsa_sparse_eager_mock_runtime(
        config(),
        (layout("cohort-0", "layer.0"),),
        device="cpu",
    )

    assert isinstance(
        runtime.coordinator.lookup_operator,
        TorchDSASparseLookupOperator,
    )


def test_producer_publishes_only_final_prefill_request_and_captures_topk():
    operator = RecordingIOOperator()
    metadata = FakeMetadata()
    main_plane = torch.zeros((4, 128, 1), dtype=torch.bfloat16)
    semantic_topk = torch.arange(
        5 * DSA_SPARSE_QUERY_WIDTH,
        dtype=torch.int32,
    ).reshape(5, DSA_SPARSE_QUERY_WIDTH)

    with begin_dsa_sparse_producer_execution(
        io_operator=operator,
        request_ids=("request-a", "request-b"),
        scheduled_token_counts=(2, 3),
        stored_token_counts=(10, 20),
        publish_requests=(False, True),
        layer_metadata={"layer.0": metadata},
        block_size=128,
    ) as context:
        assert metadata.dsa_sparse_producer_context is context
        context.publish_layer(
            "layer.0",
            (main_plane,),
            semantic_topk,
        )
        captured = context.layer_topk("layer.0")

    assert metadata.dsa_sparse_producer_context is None
    assert set(captured) == {"request-b"}
    assert captured["request-b"] == semantic_topk[-1].tolist()
    assert len(operator.publications) == 1
    publication = operator.publications[0]
    assert publication["request_transfer_id"] == "request-b"
    assert torch.equal(
        publication["block_table"],
        metadata.block_table[1],
    )
    assert publication["main_planes"][0] is main_plane


def test_consumer_initializes_topk_first_mapping_and_submits_hot_load():
    operator = RecordingIOOperator()
    runtime = create_dsa_sparse_eager_mock_runtime(
        config(),
        (layout("cohort-0", "layer.0"),),
        device="cpu",
        lookup_operator=RecordingLookupOperator(),
        io_operator=operator,
    )
    request_index = runtime.admit_mock_request("decode-request")
    topk_ids = [7, 3, 128, 7, 255]
    topk_ids.extend(
        [-1] * (DSA_SPARSE_QUERY_WIDTH - len(topk_ids))
    )
    handoff = DSASparsePDHandoff(
        remote_request_id="prefill-request",
        stored_token_count=259,
        block_size=128,
        layer_topk_by_rank={
            0: {
                "layer.0": topk_ids,
            },
        },
    )

    runtime.initialize_mock_request_from_handoff(
        "decode-request",
        handoff,
        rank=0,
    )

    expected_resident = build_dsa_sparse_resident_token_ids(
        topk_token_ids=topk_ids,
        stored_token_count=259,
        block_size=128,
    )
    cohort = runtime.coordinator.get_cohort(
        runtime.cohort_descriptors[0].cohort_key
    )
    assert cohort.state.slot_to_index[
        request_index,
        : len(expected_resident),
    ].tolist() == expected_resident
    assert cohort.state.index[request_index, 7].item() == 0
    assert cohort.state.index[request_index, 3].item() == 1

    assert len(operator.hot_initializations) == 1
    initialization = operator.hot_initializations[0]
    assert initialization["request_transfer_id"] == "prefill-request"
    assert initialization["source_token_positions"].tolist() == (
        expected_resident + [256, 257, 258]
    )
    assert initialization["destination_slots"][-3:].tolist() == [
        runtime.coordinator.config.live_tail_start + offset
        for offset in range(3)
    ]
