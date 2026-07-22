from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kv_offload.fused_indexer_kv_select import (
    CACHE_SLOTS_CAPACITY,
    OPERATOR_QUERY_HEADS,
    SELECTION_CACHE_SIZE,
    DMPFusedIndexerKVSelect,
)


class FakeFusedIndexerOps:
    def __init__(self) -> None:
        self.calls = []

    def npu_lightning_indexer_decode_update(
        self,
        query,
        key,
        weights,
        cache_slots,
        actual_seq_lengths_key,
        block_table,
    ):
        self.calls.append(
            SimpleNamespace(
                query=query,
                weights=weights,
                cache_slots=cache_slots,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
            )
        )
        batch_size = query.shape[0]
        topk_indices = torch.arange(SELECTION_CACHE_SIZE, dtype=torch.int32)
        topk_indices = topk_indices.view(1, 1, -1).expand(batch_size, 1, -1).clone()
        topk_slots = topk_indices.clone()
        miss_count = torch.zeros(batch_size, dtype=torch.int32)
        return topk_indices, topk_slots, miss_count


def make_inputs(batch_size=2, query_heads=OPERATOR_QUERY_HEADS):
    return {
        "query": torch.randn(batch_size, query_heads, 128),
        "key": torch.empty(8, 16, 1, 128),
        "weights": torch.randn(batch_size, query_heads),
        "actual_seq_lengths_key": torch.full((batch_size,), 4096, dtype=torch.int32),
        "block_table": torch.zeros(batch_size, 256, dtype=torch.int32),
    }


def test_fused_indexer_kv_select_uses_stable_per_layer_workspace():
    ops = FakeFusedIndexerOps()
    manager = DMPFusedIndexerKVSelect(
        torch.device("cpu"),
        max_microbatch_tokens=4,
        custom_ops=ops,
    )
    inputs = make_inputs()

    topk_indices = manager.select("model.layers.0.self_attn", 0, **inputs)
    workspace = manager.get_workspace("model.layers.0.self_attn", 0)
    first_cache_ptr = workspace.cache_slots.data_ptr()

    assert topk_indices.shape == (2, 1, SELECTION_CACHE_SIZE)
    assert workspace.cache_slots.shape == (4, CACHE_SLOTS_CAPACITY)
    assert workspace.cache_slots[0, :SELECTION_CACHE_SIZE].tolist() == list(range(SELECTION_CACHE_SIZE))
    assert torch.all(workspace.cache_slots[:, SELECTION_CACHE_SIZE:] == -1)
    assert workspace.topk_slots is not None
    assert workspace.miss_count is not None

    manager.select("model.layers.0.self_attn", 0, **inputs)
    assert manager.get_workspace("model.layers.0.self_attn", 0).cache_slots.data_ptr() == first_cache_ptr
    assert ops.calls[0].cache_slots.data_ptr() == ops.calls[1].cache_slots.data_ptr()


def test_fused_indexer_kv_select_separates_microbatches():
    manager = DMPFusedIndexerKVSelect(
        torch.device("cpu"),
        max_microbatch_tokens=2,
        custom_ops=FakeFusedIndexerOps(),
    )
    inputs = make_inputs()

    manager.select("model.layers.0.self_attn", 0, **inputs)
    manager.select("model.layers.0.self_attn", 1, **inputs)

    workspace_a = manager.get_workspace("model.layers.0.self_attn", 0)
    workspace_b = manager.get_workspace("model.layers.0.self_attn", 1)
    assert workspace_a.cache_slots.data_ptr() != workspace_b.cache_slots.data_ptr()


def test_fused_indexer_kv_select_zero_pads_small_test_model_heads():
    ops = FakeFusedIndexerOps()
    manager = DMPFusedIndexerKVSelect(
        torch.device("cpu"),
        max_microbatch_tokens=2,
        custom_ops=ops,
    )
    inputs = make_inputs(query_heads=4)

    manager.select("model.layers.0.self_attn", 0, **inputs)

    call = ops.calls[0]
    assert call.query.shape == (2, OPERATOR_QUERY_HEADS, 128)
    assert call.weights.shape == (2, OPERATOR_QUERY_HEADS)
    torch.testing.assert_close(call.query[:, :4], inputs["query"])
    torch.testing.assert_close(call.weights[:, :4], inputs["weights"])
    assert torch.count_nonzero(call.query[:, 4:]) == 0
    assert torch.count_nonzero(call.weights[:, 4:]) == 0


def test_fused_indexer_kv_select_rejects_oversized_microbatch():
    manager = DMPFusedIndexerKVSelect(
        torch.device("cpu"),
        max_microbatch_tokens=1,
        custom_ops=FakeFusedIndexerOps(),
    )

    with pytest.raises(RuntimeError, match="exceeds configured capacity"):
        manager.select("model.layers.0.self_attn", 0, **make_inputs())


def test_fused_indexer_kv_select_rejects_missing_operator():
    with pytest.raises(RuntimeError, match="operator is unavailable"):
        DMPFusedIndexerKVSelect(
            torch.device("cpu"),
            max_microbatch_tokens=2,
            custom_ops=SimpleNamespace(),
        )
