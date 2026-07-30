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
        "query_index": torch.zeros((2, 2048), dtype=torch.int32),
        "slot_out": torch.zeros((2, 2048), dtype=torch.int32),
        "miss_out": torch.zeros((2, 2048), dtype=torch.int32),
        "req_pool_entries": torch.tensor([0, 2], dtype=torch.int32),
        "block_table": torch.zeros((2, 8), dtype=torch.int32),
        "write_token_positions": torch.tensor(
            [130, 260],
            dtype=torch.int32,
        ),
        "write_global_slots": torch.zeros((2,), dtype=torch.int32),
        "write_destination_slots": torch.zeros((2,), dtype=torch.int32),
        "write_valid_mask": torch.ones((2,), dtype=torch.bool),
        "hot_planes": (
            torch.empty((4, 4, 1, 8), dtype=torch.bfloat16),
        ),
        "completion": object(),
    }


def test_mock_io_accepts_the_single_final_decode_abi():
    MockDSASparseIOOperator().dsa_sparse_io(**_mock_io_arguments())


def test_mock_io_leaves_miss_payload_and_descriptors_unchanged():
    arguments = _mock_io_arguments()
    arguments["miss_out"][0, 0] = 1
    arguments["slot_out"][0, 0] = 3
    arguments["hot_planes"] = (
        torch.full(
            (4, 4, 1, 8),
            7,
            dtype=torch.bfloat16,
        ),
    )
    tensor_names = (
        "query_index",
        "slot_out",
        "miss_out",
        "req_pool_entries",
        "block_table",
        "write_token_positions",
        "write_global_slots",
        "write_destination_slots",
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


def test_mock_io_requires_compact_output_shapes():
    arguments = _mock_io_arguments()
    arguments["miss_out"] = torch.zeros((1, 2048), dtype=torch.int32)

    with pytest.raises(AssertionError):
        MockDSASparseIOOperator().dsa_sparse_io(**arguments)


def test_mock_io_requires_int32_lookup_tensors():
    arguments = _mock_io_arguments()
    arguments["slot_out"] = arguments["slot_out"].to(torch.int64)

    with pytest.raises(AssertionError):
        MockDSASparseIOOperator().dsa_sparse_io(**arguments)


def test_mock_publication_and_hot_init_are_no_op_boundaries():
    operator = MockDSASparseIOOperator()
    main_plane = torch.arange(
        32,
        dtype=torch.float32,
    ).reshape(4, 8)
    block_table = torch.tensor(
        [2, 3],
        dtype=torch.int32,
    )
    main_before = main_plane.clone()
    operator.publish_main(
        context=object(),
        region=object(),
        request_transfer_id="request-a",
        block_table=block_table,
        stored_token_count=129,
        block_size=128,
        main_planes=(main_plane,),
        completion=object(),
    )

    hot_plane = torch.full(
        (1, 16, 1),
        7,
        dtype=torch.float32,
    )
    hot_before = hot_plane.clone()
    operator.initialize_hot_cache(
        context=object(),
        region=object(),
        request_transfer_id="request-a",
        request_index=0,
        source_token_positions=torch.tensor(
            [3, 9],
            dtype=torch.int32,
        ),
        destination_slots=torch.tensor(
            [0, 1],
            dtype=torch.int32,
        ),
        hot_planes=(hot_plane,),
        completion=object(),
    )

    assert torch.equal(main_plane, main_before)
    assert torch.equal(hot_plane, hot_before)
