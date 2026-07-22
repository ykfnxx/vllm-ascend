from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kv_offload.fused_indexer_kv_select import (
    CACHE_SLOTS_CAPACITY,
    OPERATOR_QUERY_HEADS,
    SELECTION_CACHE_SIZE,
    SPARSE_TOPK,
    DMPFusedIndexerKVSelect,
)


class FakeFusedIndexerOps:
    def __init__(self) -> None:
        self.calls = []
        self.gather_calls = []
        self.attention_calls = []

    def npu_lightning_indexer_decode_update_pool(
        self,
        query,
        key,
        weights,
        req_pool_entries,
        cache_slots,
        actual_seq_lengths_key,
        block_table,
    ):
        self.calls.append(
            SimpleNamespace(
                query=query,
                weights=weights,
                req_pool_entries=req_pool_entries,
                cache_slots=cache_slots,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
            )
        )
        batch_size = query.shape[0]
        topk_indices = torch.arange(SPARSE_TOPK, dtype=torch.int32)
        topk_indices = topk_indices.view(1, 1, -1).expand(batch_size, 1, -1).clone()
        topk_slots = topk_indices.clone()
        miss_count = torch.zeros(batch_size, dtype=torch.int32)
        return topk_indices, topk_slots, miss_count

    def dmp_lookup_kv_gather(self, *args, **kwargs):
        self.gather_calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return torch.zeros(args[-1].shape[0], dtype=torch.int32)

    def npu_dmp_sparse_flash_attention(self, query, *args, **kwargs):
        self.attention_calls.append(
            SimpleNamespace(query=query, args=args, kwargs=kwargs)
        )
        return torch.zeros_like(query)


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
        gather_ops=ops,
        attention_ops=ops,
    )
    inputs = make_inputs()

    topk_indices = manager.select("model.layers.0.self_attn", 0, **inputs)
    workspace = manager.get_workspace("model.layers.0.self_attn", 0)
    first_cache_ptr = ops.calls[0].cache_slots.data_ptr()

    assert topk_indices.shape == (2, 1, SPARSE_TOPK)
    assert ops.calls[0].cache_slots.shape == (8, CACHE_SLOTS_CAPACITY)
    assert ops.calls[0].cache_slots[0, :SELECTION_CACHE_SIZE].tolist() == list(
        range(SELECTION_CACHE_SIZE)
    )
    assert torch.all(ops.calls[0].cache_slots[:, SELECTION_CACHE_SIZE:] == -1)
    assert workspace.req_pool_entries.tolist() == [0, 1, 2, 3]
    assert workspace.topk_slots is not None
    assert workspace.miss_count is not None

    manager.select("model.layers.0.self_attn", 0, **inputs)
    assert ops.calls[0].cache_slots.data_ptr() == ops.calls[1].cache_slots.data_ptr()
    assert ops.calls[1].cache_slots.data_ptr() == first_cache_ptr


def test_fused_indexer_kv_select_separates_microbatches():
    ops = FakeFusedIndexerOps()
    manager = DMPFusedIndexerKVSelect(
        torch.device("cpu"),
        max_microbatch_tokens=2,
        custom_ops=ops,
        gather_ops=ops,
        attention_ops=ops,
    )
    inputs = make_inputs()

    manager.select("model.layers.0.self_attn", 0, **inputs)
    manager.select("model.layers.0.self_attn", 1, **inputs)

    workspace_a = manager.get_workspace("model.layers.0.self_attn", 0)
    workspace_b = manager.get_workspace("model.layers.0.self_attn", 1)
    assert ops.calls[0].cache_slots.data_ptr() == ops.calls[1].cache_slots.data_ptr()
    assert workspace_a.req_pool_entries.tolist() == [0, 1]
    assert workspace_b.req_pool_entries.tolist() == [2, 3]
    assert set(workspace_a.req_pool_entries.tolist()).isdisjoint(
        workspace_b.req_pool_entries.tolist()
    )


def test_fused_indexer_kv_select_zero_pads_small_test_model_heads():
    ops = FakeFusedIndexerOps()
    manager = DMPFusedIndexerKVSelect(
        torch.device("cpu"),
        max_microbatch_tokens=2,
        custom_ops=ops,
        gather_ops=ops,
        attention_ops=ops,
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


def test_fused_indexer_kv_select_runs_one_kvio_then_selected_cache_sfa():
    ops = FakeFusedIndexerOps()
    manager = DMPFusedIndexerKVSelect(
        torch.device("cpu"),
        max_microbatch_tokens=2,
        custom_ops=ops,
        gather_ops=ops,
        attention_ops=ops,
        block_size=128,
    )
    inputs = make_inputs()
    topk_indices = manager.select("model.layers.0.self_attn", 0, **inputs)
    ql_nope = torch.randn(2, 4, 64)
    q_pe = torch.randn(2, 4, 64)
    output = torch.empty(2, 4, 64)
    indexer_result = (ql_nope, q_pe, topk_indices, output)
    kv_cache = (
        torch.randn(512, 128, 1, 64),
        torch.randn(512, 128, 1, 64),
        inputs["key"],
    )
    metadata = SimpleNamespace(
        seq_lens=inputs["actual_seq_lengths_key"],
        block_table=torch.zeros(2, 512, dtype=torch.int32),
        cum_query_lens=torch.tensor([1, 2], dtype=torch.int32),
    )

    manager.prepare_attention(
        "model.layers.0.self_attn", 0, indexer_result, kv_cache, metadata
    )
    manager.gather("model.layers.0.self_attn", 0, kv_cache, metadata)
    attention_output = manager.run_attention(
        "model.layers.0.self_attn", 0, indexer_result, 0.125, metadata
    )

    assert len(ops.gather_calls) == 1
    gather_args = ops.gather_calls[0].args
    assert gather_args[4].shape == (2, SPARSE_TOPK)
    assert gather_args[5].shape == (2, SPARSE_TOPK)
    assert torch.all(gather_args[6] == 1)
    assert len(ops.attention_calls) == 1
    sparse_indices = ops.attention_calls[0].args[2]
    torch.testing.assert_close(sparse_indices.squeeze(1), gather_args[5])
    assert attention_output.shape == ql_nope.shape


def test_fused_indexer_kv_select_rejects_oversized_microbatch():
    ops = FakeFusedIndexerOps()
    manager = DMPFusedIndexerKVSelect(
        torch.device("cpu"),
        max_microbatch_tokens=1,
        custom_ops=ops,
        gather_ops=ops,
        attention_ops=ops,
    )

    with pytest.raises(RuntimeError, match="exceeds configured capacity"):
        manager.select("model.layers.0.self_attn", 0, **make_inputs())


def test_fused_indexer_kv_select_rejects_missing_operator():
    with pytest.raises(RuntimeError, match="operator is unavailable"):
        DMPFusedIndexerKVSelect(
            torch.device("cpu"),
            max_microbatch_tokens=2,
            custom_ops=SimpleNamespace(),
            gather_ops=SimpleNamespace(dmp_lookup_kv_gather=lambda *args: None),
            attention_ops=SimpleNamespace(
                npu_dmp_sparse_flash_attention=lambda *args: None
            ),
        )
