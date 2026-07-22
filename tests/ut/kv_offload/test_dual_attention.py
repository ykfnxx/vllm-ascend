from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kv_offload.dual_attention import DMPDualAttention


class FakeDualAttentionOps:
    def __init__(self) -> None:
        self.select_calls = []
        self.gather_calls = []
        self.sfa_calls = []
        self.merge_calls = []

    def npu_kv_select_out(self, *args, **kwargs):
        self.select_calls.append((args, kwargs))
        args[10].fill_(-1)
        args[10][..., 0] = 0
        args[11].fill_(-1)
        args[11][..., 1:] = 1
        args[12].fill_(-1)
        args[12][..., 1:] = 1
        args[13].fill_(1)
        args[14].fill_(3)
        args[15].fill_(3)
        args[16].fill_(1)
        args[17].zero_()

    def npu_kv_gather_out(self, *args, **kwargs):
        self.gather_calls.append((args, kwargs))
        args[17].fill_(4)

    def npu_dmp_sparse_flash_attention(self, *args, **kwargs):
        self.sfa_calls.append((args, kwargs))
        kwargs["softmax_max_out"].zero_()
        kwargs["softmax_sum_out"].fill_(1)
        return (args[0].clone(),)

    def npu_da_attention_merge(self, *args):
        self.merge_calls.append(args)
        return (args[0] + args[3],)


class FakeAttentionOnlyOps:
    def npu_dmp_sparse_flash_attention(self, *args, **kwargs):
        kwargs["softmax_max_out"].zero_()
        kwargs["softmax_sum_out"].fill_(1)
        return (args[0].clone(),)

    def npu_da_attention_merge(self, *args):
        return (args[0] + args[3],)


class FakeHixlBackend:
    pool_size = 8
    cache_size = 8

    def __init__(self) -> None:
        self.select_calls = []
        self.gather_calls = []
        self.prepared = []
        self._cache = {}

    def cache_tensors(self, layer_name, kv_dtype, rope_dtype):
        if layer_name not in self._cache:
            self._cache[layer_name] = (
                torch.empty(16, 4, 6, dtype=kv_dtype),
                torch.empty(16, 4, 2, dtype=rope_dtype),
            )
        return self._cache[layer_name]

    def prepare_workspace(self, layer_name, workspace, microbatch_idx):
        state = SimpleNamespace(
            layer_name=layer_name,
            microbatch_idx=microbatch_idx,
        )
        self.prepared.append(state)
        return state

    def select(
        self,
        layer_name,
        microbatch_idx,
        workspace,
        topk_indices,
        attn_metadata,
    ):
        self.select_calls.append((layer_name, microbatch_idx, topk_indices))
        workspace.hit_sparse_indices.fill_(-1)
        workspace.hit_sparse_indices[..., 0] = 0
        workspace.miss_insert_indices.fill_(-1)
        workspace.miss_insert_indices[..., 1:] = 1
        workspace.selected_actual_seq.fill_(self.cache_size)
        workspace.selection_kv_actual_seq.fill_(self.cache_size)

    def gather(self, workspace):
        self.gather_calls.append(workspace.backend_state)


def make_runtime():
    ops = FakeDualAttentionOps()
    manager = DMPDualAttention(
        torch.device("cpu"),
        block_size=4,
        custom_ops=ops,
        select_stream=object(),
        gather_stream=object(),
        max_microbatch_tokens=4,
    )
    ql_nope = torch.randn(2, 3, 6)
    q_pe = torch.randn(2, 3, 2)
    topk_indices = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.int32)
    output = torch.empty(2, 8)
    indexer_result = (ql_nope, q_pe, topk_indices, output)
    kv_cache = (
        torch.randn(8, 4, 1, 6),
        torch.randn(8, 4, 1, 2),
    )
    metadata = SimpleNamespace(
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        cum_query_lens=torch.tensor([1, 2], dtype=torch.int32),
    )
    return manager, ops, indexer_result, kv_cache, metadata


def test_dual_attention_runs_split_pipeline_with_stable_workspace():
    manager, ops, indexer_result, kv_cache, metadata = make_runtime()

    manager.select("layer.0", 0, indexer_result, kv_cache, metadata)
    workspace = manager.get_workspace("layer.0", 0)

    assert workspace.hit_sparse_indices.shape == (2, 1, 4)
    assert workspace.selection_kv_block_table.shape == (2, 1)
    assert workspace.selected_actual_seq.tolist() == [4, 4]

    manager.run_hit_attention("layer.0", 0, indexer_result, 0.125, metadata)
    manager.gather("layer.0", 0, kv_cache, metadata)
    output = manager.run_miss_attention_and_merge("layer.0", 0, indexer_result, 0.125, metadata)

    assert output.shape == indexer_result[0].shape
    assert len(ops.select_calls) == 1
    assert len(ops.gather_calls) == 1
    assert len(ops.sfa_calls) == 2
    assert len(ops.merge_calls) == 1
    assert ops.sfa_calls[0][1]["actual_seq_lengths_kv"] is workspace.selected_actual_seq
    assert ops.sfa_calls[1][1]["actual_seq_lengths_kv"] is workspace.selection_kv_actual_seq

    manager.select("layer.0", 0, indexer_result, kv_cache, metadata)
    assert manager.get_workspace("layer.0", 0) is workspace


def test_dual_attention_keeps_separate_graph_shapes():
    manager, _, indexer_result, kv_cache, metadata = make_runtime()
    manager.select("layer.0", 0, indexer_result, kv_cache, metadata)
    small_workspace = manager.get_workspace("layer.0", 0)

    large_result = tuple(torch.cat([tensor, tensor], dim=0) for tensor in indexer_result)
    large_metadata = SimpleNamespace(
        block_table=torch.cat([metadata.block_table, metadata.block_table + 4]),
        seq_lens=torch.cat([metadata.seq_lens, metadata.seq_lens]),
        cum_query_lens=torch.arange(1, 5, dtype=torch.int32),
    )
    manager.select("layer.0", 0, large_result, kv_cache, large_metadata)
    large_workspace = manager.get_workspace("layer.0", 0)

    assert large_workspace is not small_workspace
    assert large_workspace.hit_sparse_indices.shape == (4, 1, 4)
    assert large_workspace.selection_kv_cache.data_ptr() == small_workspace.selection_kv_cache.data_ptr()


def test_dual_attention_resets_pool_when_request_slot_is_reassigned():
    manager, _, indexer_result, kv_cache, metadata = make_runtime()
    manager.select("layer.0", 1, indexer_result, kv_cache, metadata)
    workspace = manager.get_workspace("layer.0", 1)
    workspace.selection_kv_block_status.fill_(7)

    metadata.seq_lens.add_(1)
    manager.select("layer.0", 1, indexer_result, kv_cache, metadata)
    assert torch.all(workspace.selection_kv_block_status == 7)

    metadata.seq_lens.fill_(1)
    manager.select("layer.0", 1, indexer_result, kv_cache, metadata)
    assert torch.all(workspace.selection_kv_block_status == -1)


def test_dual_attention_rejects_missing_custom_operator():
    class IncompleteOps:
        npu_kv_select_out = object()

    with pytest.raises(RuntimeError, match="npu_kv_gather_out"):
        DMPDualAttention(
            torch.device("cpu"),
            block_size=4,
            custom_ops=IncompleteOps(),
            select_stream=object(),
            gather_stream=object(),
        )


def test_hixl_backend_keeps_local_backend_optional_and_partitions_cache():
    hixl = FakeHixlBackend()
    manager = DMPDualAttention(
        torch.device("cpu"),
        block_size=4,
        custom_ops=FakeAttentionOnlyOps(),
        select_stream=object(),
        gather_stream=object(),
        max_microbatch_tokens=4,
        stream_mode="two",
        kv_backend="hixl",
        hixl_backend=hixl,
    )
    _, _, indexer_result, kv_cache, metadata = make_runtime()

    manager.select("model.layers.0.self_attn", 0, indexer_result, kv_cache, metadata)
    manager.select("model.layers.0.self_attn", 1, indexer_result, kv_cache, metadata)
    workspace_a = manager.get_workspace("model.layers.0.self_attn", 0)
    workspace_b = manager.get_workspace("model.layers.0.self_attn", 1)

    assert workspace_a.selection_kv_block_table.tolist() == [[0, 1], [2, 3]]
    assert workspace_b.selection_kv_block_table.tolist() == [[0, 1], [2, 3]]
    assert workspace_a.selection_kv_cache.data_ptr() != workspace_b.selection_kv_cache.data_ptr()
    assert workspace_a.backend_state.microbatch_idx == 0
    assert workspace_b.backend_state.microbatch_idx == 1

    manager.gather("model.layers.0.self_attn", 1, kv_cache, metadata)
    assert hixl.gather_calls == [workspace_b.backend_state]


def test_dual_attention_rejects_invalid_kv_backend():
    with pytest.raises(ValueError, match="Unsupported DMP KV backend"):
        DMPDualAttention(
            torch.device("cpu"),
            block_size=4,
            custom_ops=FakeDualAttentionOps(),
            select_stream=object(),
            gather_stream=object(),
            kv_backend="unknown",
        )


def test_hixl_backend_rejects_four_stream_workspace_race():
    with pytest.raises(ValueError, match="requires two-stream mode"):
        DMPDualAttention(
            torch.device("cpu"),
            block_size=4,
            custom_ops=FakeAttentionOnlyOps(),
            select_stream=object(),
            gather_stream=object(),
            max_microbatch_tokens=4,
            stream_mode="four",
            kv_backend="hixl",
            hixl_backend=FakeHixlBackend(),
        )


def test_dual_attention_two_stream_mode_shares_auxiliary_stream():
    auxiliary_stream = object()
    manager = DMPDualAttention(
        torch.device("cpu"),
        block_size=4,
        custom_ops=FakeDualAttentionOps(),
        select_stream=auxiliary_stream,
        stream_mode="two",
    )

    assert manager.stream_mode == "two"
    assert manager.select_stream is auxiliary_stream
    assert manager.gather_stream is auxiliary_stream
    main_stream = object()
    assert manager.get_indexer_a_stream(main_stream, auxiliary_stream) is main_stream


def test_dual_attention_four_stream_mode_keeps_separate_streams():
    select_stream = object()
    gather_stream = object()
    manager = DMPDualAttention(
        torch.device("cpu"),
        block_size=4,
        custom_ops=FakeDualAttentionOps(),
        select_stream=select_stream,
        gather_stream=gather_stream,
        stream_mode="4",
    )

    assert manager.stream_mode == "four"
    assert manager.select_stream is select_stream
    assert manager.gather_stream is gather_stream
    main_stream = object()
    dmp_stream = object()
    assert manager.get_indexer_a_stream(main_stream, dmp_stream) is dmp_stream


def test_dual_attention_rejects_invalid_stream_mode():
    with pytest.raises(ValueError, match="Unsupported DMP stream mode"):
        DMPDualAttention(
            torch.device("cpu"),
            block_size=4,
            custom_ops=FakeDualAttentionOps(),
            select_stream=object(),
            gather_stream=object(),
            stream_mode="three",
        )


def test_dual_attention_requires_one_selection_head():
    manager, _, indexer_result, kv_cache, metadata = make_runtime()
    bad_result = (
        indexer_result[0],
        indexer_result[1],
        torch.zeros(2, 2, 4, dtype=torch.int32),
        indexer_result[3],
    )

    with pytest.raises(ValueError, match="one selection head"):
        manager.select("layer.0", 0, bad_result, kv_cache, metadata)
