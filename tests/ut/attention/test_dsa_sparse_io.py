# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest

from vllm_ascend.attention.dsa_sparse_io import (
    DSASparseIOBackendRegistry,
    DSASparseRegionKey,
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


@pytest.mark.parametrize(
    ("method_name", "expected_message"),
    [
        ("publish_async", "publish operator"),
        ("wait_publish", "publish wait"),
        ("read_async", "read operator"),
        ("wait_read", "read wait"),
        ("write_async", "write operator"),
        ("wait_write", "write wait"),
    ],
)
def test_unimplemented_io_operator_is_an_explicit_stub(
    method_name,
    expected_message,
):
    operator = UnimplementedDSASparseIOOperator()

    with pytest.raises(NotImplementedError, match=expected_message):
        getattr(operator, method_name)()
