# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.attention.dsa_sparse_io import (
    DSASparseIOBackendRegistry,
    DSASparseRegionKey,
    MockDSASparseIOOperator,
    UnimplementedDSASparseIOOperator,
)


class FakeBackend:
    def __init__(self, options):
        self.options = options
        self.frozen = False

    def freeze(self):
        self.frozen = True


def test_backend_registry_is_explicit_and_freezes():
    registry = DSASparseIOBackendRegistry()
    registry.register("fake", FakeBackend)

    backend = registry.create("fake", {"namespace": "test"})
    assert backend.options == {"namespace": "test"}

    registry.freeze()
    assert backend.frozen
    with pytest.raises(RuntimeError, match="registry is frozen"):
        registry.register("late", lambda options: options)
    with pytest.raises(RuntimeError, match="registry is frozen"):
        registry.create("fake", {})


def test_backend_registry_rejects_duplicate_names():
    registry = DSASparseIOBackendRegistry()
    registry.register("fake", lambda options: options)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("fake", lambda options: options)


def test_region_key_isolates_target_and_draft_graphs():
    common = {
        "deployment_id": "deployment",
        "instance_id": "decode-0",
        "kv_role": "kv_consumer",
        "pp_rank": 0,
        "tp_rank": 3,
        "layer_name": "model.layers.0.self_attn",
    }

    target = DSASparseRegionKey(graph_role="target", **common)
    draft = DSASparseRegionKey(graph_role="draft", **common)

    assert target != draft


def test_unimplemented_io_operator_is_an_explicit_stub():
    operator = UnimplementedDSASparseIOOperator()

    with pytest.raises(NotImplementedError, match="unified I/O"):
        operator.dsa_sparse_io()


def _mock_io_arguments():
    return {
        "context": object(),
        "region": object(),
        "topk_positions": torch.zeros((2, 4), dtype=torch.int32),
        "resolved_hot_indices": torch.zeros((2, 4), dtype=torch.int32),
        "miss_mask": torch.zeros((2, 4), dtype=torch.bool),
        "query_to_req_idx": torch.tensor([0, 1], dtype=torch.int32),
        "block_table": torch.zeros((2, 8), dtype=torch.int32),
        "write_global_slots": torch.zeros((2, 1), dtype=torch.int32),
        "write_destination_hot_row_ids": torch.zeros(
            (2, 1),
            dtype=torch.int32,
        ),
        "write_valid_mask": torch.ones((2, 1), dtype=torch.bool),
        "hot_planes": (
            torch.empty((4, 4, 1, 8), dtype=torch.bfloat16),
        ),
        "completion": object(),
    }


def test_mock_io_accepts_the_single_final_decode_abi():
    MockDSASparseIOOperator().dsa_sparse_io(**_mock_io_arguments())


def test_mock_io_leaves_miss_payload_and_descriptors_unchanged():
    arguments = _mock_io_arguments()
    arguments["miss_mask"][0, 0] = True
    arguments["resolved_hot_indices"][0, 0] = 3
    arguments["hot_planes"] = (
        torch.full(
            (4, 4, 1, 8),
            7,
            dtype=torch.bfloat16,
        ),
    )
    tensor_names = (
        "topk_positions",
        "resolved_hot_indices",
        "miss_mask",
        "block_table",
        "write_global_slots",
        "write_destination_hot_row_ids",
        "write_valid_mask",
    )
    before = {
        name: arguments[name].clone()
        for name in tensor_names
    }
    hot_before = tuple(
        plane.clone()
        for plane in arguments["hot_planes"]
    )

    MockDSASparseIOOperator().dsa_sparse_io(**arguments)

    for name, expected in before.items():
        assert torch.equal(arguments[name], expected)
    for plane, expected in zip(arguments["hot_planes"], hot_before):
        assert torch.equal(plane, expected)


def test_mock_io_validates_shapes_without_reading_tensor_values():
    arguments = _mock_io_arguments()
    arguments["miss_mask"] = torch.zeros((1, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match="Top-K tensor shape"):
        MockDSASparseIOOperator().dsa_sparse_io(**arguments)


def test_mock_io_rejects_non_int32_block_table():
    arguments = _mock_io_arguments()
    arguments["block_table"] = torch.zeros(
        (2, 8),
        dtype=torch.int64,
    )

    with pytest.raises(TypeError, match="block_table must use int32"):
        MockDSASparseIOOperator().dsa_sparse_io(**arguments)
