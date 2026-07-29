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
from vllm_ascend.dsa_sparse_constants import DSA_SPARSE_QUERY_WIDTH
from vllm_ascend.worker.dsa_sparse_eager import (
    DSASparseEagerCohortLayout,
    create_dsa_sparse_eager_mock_runtime,
)


@dataclass
class FakeMetadata:
    num_input_tokens: int = 2
    num_actual_tokens: int = 2
    seq_lens: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    dsa_sparse_context: object | None = None

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


def test_default_runtime_lookup_boundary_is_explicitly_unimplemented():
    runtime = create_dsa_sparse_eager_mock_runtime(
        config(),
        (layout("cohort-0", "layer.0"),),
        device="cpu",
    )
    runtime.admit_mock_request("request-a")
    metadata = FakeMetadata(
        num_input_tokens=1,
        num_actual_tokens=1,
        seq_lens=torch.tensor([131], dtype=torch.int32),
        block_table=torch.tensor(
            [[10, 11, 12, 13]],
            dtype=torch.int32,
        ),
    )

    execution = runtime.begin_target_batch(
        request_ids=("request-a",),
        query_positions=torch.tensor([130]),
        layer_metadata={"layer.0": metadata},
    )
    with pytest.raises(NotImplementedError, match="not implemented"):
        with execution as router:
            router.main_write_target("layer.0")
            router.submit_newest_write("layer.0")
            router.run_layer_attention(
                "layer.0",
                topk()[:1],
                lambda resolution: torch.empty(0),
            )
