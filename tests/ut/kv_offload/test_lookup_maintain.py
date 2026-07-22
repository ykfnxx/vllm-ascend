from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kv_offload.lookup_maintain import (
    FIXED_MISS_COUNT,
    FREE_HEAD_STRIDE,
    INDEX_CAPACITY,
    QUERY_SLOT_COUNT,
    RESIDENT_SLOT_COUNT,
    TOTAL_SLOT_COUNT,
    DMPLookupMaintain,
)


class FakeLookupMaintainOps:
    def __init__(self):
        self.lookup_calls = []
        self.maintain_calls = []
        self.slot_out = None
        self.miss_out = None
        self.gather_calls = []

    def asu_hbm_index_lookup(
        self,
        token_to_slot,
        slot_to_token,
        free_slots,
        free_head,
        pool_entries,
        query_index,
        seq_lens,
        needs_refill,
        req_num,
    ):
        self.lookup_calls.append(
            SimpleNamespace(
                state=token_to_slot,
                pool_entries=pool_entries.clone(),
                query_index=query_index.clone(),
                seq_lens=seq_lens.clone(),
                needs_refill=needs_refill.clone(),
                req_num=req_num,
            )
        )
        slot_out = (
            query_index.clone() if self.slot_out is None else self.slot_out.clone()
        )
        miss_out = (
            torch.zeros_like(query_index)
            if self.miss_out is None
            else self.miss_out.clone()
        )
        valid = (query_index >= 0) & (query_index < seq_lens.view(-1, 1))
        misses = valid & (miss_out != 0)
        hits = valid & ~misses
        hit_sparse = torch.where(hits, query_index, -1)
        miss_sparse = torch.where(misses, slot_out, -1)
        resident_token_ids = slot_to_token[:req_num].clone()
        return slot_out, miss_out, hit_sparse, miss_sparse, resident_token_ids

    def dmp_lookup_kv_gather(self, *args, **kwargs):
        self.gather_calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return torch.ones(args[4].shape[0], dtype=torch.int32, device=args[4].device)

    def asu_hbm_index_maintain_aicpu(
        self,
        token_to_slot,
        slot_to_token,
        free_slots,
        free_head,
        pool_entries,
        slot_out,
        req_num,
        seed,
    ):
        self.maintain_calls.append(
            SimpleNamespace(
                state=token_to_slot,
                pool_entries=pool_entries.clone(),
                slot_out=slot_out.clone(),
                req_num=req_num,
                seed=seed,
            )
        )


class FakeAttentionOps:
    def __init__(self):
        self.sfa_calls = []
        self.merge_calls = []

    def npu_dmp_sparse_flash_attention(self, *args, **kwargs):
        self.sfa_calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        kwargs["softmax_max_out"].zero_()
        kwargs["softmax_sum_out"].fill_(1)
        return args[0].clone()

    def npu_da_attention_merge(self, *args, **kwargs):
        self.merge_calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return args[0] + args[3]


def make_manager(*, num_layers=2, max_microbatch_tokens=3):
    index_ops = FakeLookupMaintainOps()
    attention_ops = FakeAttentionOps()
    manager = DMPLookupMaintain(
        torch.device("cpu"),
        num_layers=num_layers,
        max_microbatch_tokens=max_microbatch_tokens,
        max_model_len=132000,
        block_size=128,
        custom_ops=index_ops,
        attention_ops=attention_ops,
        maintain_stream=object(),
    )
    return manager, index_ops, attention_ops


def make_runtime_inputs(batch_size=1, seq_len=12000):
    ql_nope = torch.zeros((batch_size, 1, 8), dtype=torch.float16)
    q_pe = torch.zeros((batch_size, 1, 4), dtype=torch.float16)
    topk = torch.arange(QUERY_SLOT_COUNT, dtype=torch.int32).view(1, 1, -1)
    topk = topk.expand(batch_size, -1, -1).clone()
    output = torch.empty((batch_size, 8), dtype=torch.float16)
    kv_cache = (
        torch.zeros((256, 128, 8), dtype=torch.float16),
        torch.zeros((256, 128, 4), dtype=torch.float16),
    )
    metadata = SimpleNamespace(
        block_table=torch.arange(256, dtype=torch.int32)
        .view(1, -1)
        .expand(batch_size, -1)
        .clone(),
        seq_lens=torch.full((batch_size,), seq_len, dtype=torch.int32),
        cum_query_lens=torch.arange(1, batch_size + 1, dtype=torch.int32),
    )
    return (ql_nope, q_pe, topk, output), kv_cache, metadata


def test_lookup_maintain_preallocates_graph_stable_state():
    manager, _, _ = make_manager()

    assert len(manager._workspaces) == 4
    workspace = manager.get_workspace(0, 0)
    assert workspace.token_to_slot.shape == (3, INDEX_CAPACITY)
    assert workspace.slot_to_token.shape == (3, TOTAL_SLOT_COUNT)
    assert workspace.free_slots.shape == (3, QUERY_SLOT_COUNT)
    assert workspace.free_head.shape == (3, FREE_HEAD_STRIDE)
    torch.testing.assert_close(
        workspace.token_to_slot[0, :RESIDENT_SLOT_COUNT],
        torch.arange(RESIDENT_SLOT_COUNT, dtype=torch.int32),
    )


def test_lookup_maintain_keeps_layer_and_microbatch_state_separate():
    manager, ops, _ = make_manager()
    topk = torch.arange(QUERY_SLOT_COUNT, dtype=torch.int32).view(1, 1, -1)

    manager.update(layer_idx=0, microbatch_idx=0, topk_indices=topk)
    manager.update(layer_idx=1, microbatch_idx=1, topk_indices=topk)

    assert ops.lookup_calls[0].state is manager.get_workspace(0, 0).token_to_slot
    assert ops.lookup_calls[1].state is manager.get_workspace(1, 1).token_to_slot
    assert ops.lookup_calls[0].state is not ops.lookup_calls[1].state
    assert ops.maintain_calls[0].seed == 0
    assert ops.maintain_calls[1].seed == 3


def test_lookup_and_maintain_are_independently_schedulable():
    manager, ops, _ = make_manager(num_layers=1)
    indexer_result, kv_cache, metadata = make_runtime_inputs()

    manager.lookup(
        layer_idx=0,
        microbatch_idx=0,
        indexer_result=indexer_result,
        kv_cache=kv_cache,
        attn_metadata=metadata,
    )
    assert len(ops.lookup_calls) == 1
    assert not ops.maintain_calls

    manager.maintain(layer_idx=0, microbatch_idx=0)
    assert len(ops.maintain_calls) == 1


def test_first_request_uses_one_miss_only_gather_call():
    manager, index_ops, _ = make_manager(num_layers=1, max_microbatch_tokens=1)
    indexer_result, kv_cache, metadata = make_runtime_inputs()

    manager.lookup(
        layer_idx=0,
        microbatch_idx=0,
        indexer_result=indexer_result,
        kv_cache=kv_cache,
        attn_metadata=metadata,
    )
    workspace = manager.get_attention_workspace(0, 0)
    assert workspace.needs_refill.tolist() == [True]

    manager.gather(
        layer_idx=0,
        microbatch_idx=0,
        kv_cache=kv_cache,
        attn_metadata=metadata,
    )
    assert len(index_ops.gather_calls) == 1
    call = index_ops.gather_calls[0]
    assert call.args[2].shape == (1, 80)
    assert call.args[3].shape == (1, TOTAL_SLOT_COUNT)
    assert call.args[7].tolist() == [True]


def test_misses_keep_global_10k_slots_for_one_gather_call():
    manager, index_ops, _ = make_manager(num_layers=1, max_microbatch_tokens=1)
    indexer_result, kv_cache, metadata = make_runtime_inputs()
    query = indexer_result[2].reshape(1, -1)
    slots = torch.arange(QUERY_SLOT_COUNT, dtype=torch.int32).view(1, -1)
    bank_slots = [3, 2050, 4097, 6148, 8197]
    slots[0, : len(bank_slots)] = torch.tensor(bank_slots, dtype=torch.int32)
    misses = torch.zeros_like(slots)
    misses[0, : len(bank_slots)] = 1
    index_ops.slot_out = slots
    index_ops.miss_out = misses

    index_workspace = manager.get_workspace(0, 0)
    index_workspace.request_signature[0] = metadata.block_table[0, 0]
    index_workspace.previous_seq_lens[0] = metadata.seq_lens[0] - 1
    manager.lookup(
        layer_idx=0,
        microbatch_idx=0,
        indexer_result=indexer_result,
        kv_cache=kv_cache,
        attn_metadata=metadata,
    )
    workspace = manager.get_attention_workspace(0, 0)
    assert workspace.needs_refill.tolist() == [False]

    assert workspace.miss_sparse_indices[0, 0, :5].tolist() == bank_slots

    manager.gather(
        layer_idx=0,
        microbatch_idx=0,
        kv_cache=kv_cache,
        attn_metadata=metadata,
    )
    assert len(index_ops.gather_calls) == 1
    call = index_ops.gather_calls[0]
    torch.testing.assert_close(call.args[4], query)
    torch.testing.assert_close(call.args[5], slots)
    torch.testing.assert_close(call.args[6], misses)
    assert call.args[2].shape == (1, 80)
    assert call.args[2].max() < 80


def test_segmented_sfa_uses_global_10k_slots_and_merges():
    manager, index_ops, attention_ops = make_manager(
        num_layers=1, max_microbatch_tokens=1
    )
    indexer_result, kv_cache, metadata = make_runtime_inputs()
    slots = torch.arange(QUERY_SLOT_COUNT, dtype=torch.int32).view(1, -1)
    slots[0, 0] = 9000
    index_ops.slot_out = slots
    index_ops.miss_out = torch.zeros_like(slots)
    index_ops.miss_out[0, 0] = 1
    index_workspace = manager.get_workspace(0, 0)
    index_workspace.request_signature[0] = metadata.block_table[0, 0]
    index_workspace.previous_seq_lens[0] = metadata.seq_lens[0] - 1

    manager.lookup(
        layer_idx=0,
        microbatch_idx=0,
        indexer_result=indexer_result,
        kv_cache=kv_cache,
        attn_metadata=metadata,
    )
    manager.run_hit_attention(
        layer_idx=0,
        microbatch_idx=0,
        indexer_result=indexer_result,
        scale=1.0,
        attn_metadata=metadata,
    )
    result = manager.run_miss_attention_and_merge(
        layer_idx=0,
        microbatch_idx=0,
        indexer_result=indexer_result,
        scale=1.0,
        attn_metadata=metadata,
    )

    assert len(attention_ops.sfa_calls) == 2
    assert attention_ops.sfa_calls[1].args[3][0, 0, 0] == 9000
    assert attention_ops.sfa_calls[0].kwargs["block_table"].shape == (1, 256)
    assert attention_ops.sfa_calls[0].kwargs["actual_seq_lengths_kv"].tolist() == [
        12000
    ]
    assert len(attention_ops.merge_calls) == 1
    assert result.shape == indexer_result[0].shape


def test_combined_sfa_merges_microbatch_indexer_results_once():
    manager, _, attention_ops = make_manager(
        num_layers=1, max_microbatch_tokens=1
    )
    indexer_a, kv_cache, metadata_a = make_runtime_inputs()
    indexer_b, _, metadata_b = make_runtime_inputs()
    indexer_b = (
        torch.ones_like(indexer_b[0]),
        torch.ones_like(indexer_b[1]),
        indexer_b[2],
        indexer_b[3],
    )
    for microbatch_idx, indexer_result, metadata in (
        (0, indexer_a, metadata_a),
        (1, indexer_b, metadata_b),
    ):
        manager.lookup(
            layer_idx=0,
            microbatch_idx=microbatch_idx,
            indexer_result=indexer_result,
            kv_cache=kv_cache,
            attn_metadata=metadata,
        )

    workspace_a = manager.get_attention_workspace(0, 0)
    workspace_b = manager.get_attention_workspace(0, 1)
    assert (
        workspace_a.selection_kv_cache.untyped_storage().data_ptr()
        == workspace_b.selection_kv_cache.untyped_storage().data_ptr()
    )
    prepared_inputs = manager.prepare_combined_attention(
        layer_idx=0,
        indexer_results=(indexer_a, indexer_b),
        attn_metadata=(metadata_a, metadata_b),
    )
    assert len(attention_ops.sfa_calls) == 0
    manager.run_combined_hit_attention(
        layer_idx=0,
        indexer_results=(indexer_a, indexer_b),
        scale=1.0,
        attn_metadata=(metadata_a, metadata_b),
        prepared_inputs=prepared_inputs,
    )
    assert len(attention_ops.sfa_calls) == 1
    assert len(attention_ops.merge_calls) == 0
    first_call = attention_ops.sfa_calls[0]
    assert first_call.kwargs["block_table"].shape == (2, 256)
    assert first_call.kwargs["actual_seq_lengths_kv"].tolist() == [12000, 12000]

    result = manager.run_combined_miss_attention_and_merge(
        layer_idx=0,
        indexer_results=(indexer_a, indexer_b),
        scale=1.0,
        attn_metadata=(metadata_a, metadata_b),
        prepared_inputs=prepared_inputs,
    )

    assert FIXED_MISS_COUNT == 300
    assert len(attention_ops.sfa_calls) == 2
    assert len(attention_ops.merge_calls) == 1
    assert first_call.args[0].shape == (2, 1, 8)
    assert first_call.args[3].shape == (2, 1, QUERY_SLOT_COUNT)
    assert first_call.kwargs["actual_seq_lengths_query"].tolist() == [1, 2]
    assert attention_ops.sfa_calls[1].kwargs["block_table"].shape == (2, 80)
    assert result.shape == (2, 1, 8)


def test_lookup_kernel_receives_unclamped_indices_for_validity_filtering():
    manager, ops, _ = make_manager()
    topk = torch.arange(QUERY_SLOT_COUNT, dtype=torch.int64).view(1, -1)
    topk[0, 0] = -1
    topk[0, 1] = INDEX_CAPACITY + 10
    original = topk.clone()

    manager.update(layer_idx=0, microbatch_idx=0, topk_indices=topk)

    torch.testing.assert_close(topk, original)
    assert ops.lookup_calls[0].query_index.dtype == torch.int32
    assert ops.lookup_calls[0].query_index[0, 0] == -1
    assert ops.lookup_calls[0].query_index[0, 1] == INDEX_CAPACITY + 10


def test_lookup_maintain_rejects_incompatible_shapes_and_capacity():
    manager, _, _ = make_manager(max_microbatch_tokens=1)
    wrong_width = torch.zeros((1, QUERY_SLOT_COUNT - 1), dtype=torch.int32)
    with pytest.raises(RuntimeError, match="TopK width 2048"):
        manager.update(layer_idx=0, microbatch_idx=0, topk_indices=wrong_width)

    too_large = torch.zeros((2, QUERY_SLOT_COUNT), dtype=torch.int32)
    with pytest.raises(RuntimeError, match="exceeds configured capacity"):
        manager.update(layer_idx=0, microbatch_idx=0, topk_indices=too_large)

    with pytest.raises(ValueError, match="index capacity"):
        DMPLookupMaintain(
            torch.device("cpu"),
            num_layers=1,
            max_microbatch_tokens=1,
            max_model_len=INDEX_CAPACITY + 1,
            custom_ops=FakeLookupMaintainOps(),
            attention_ops=FakeAttentionOps(),
            maintain_stream=object(),
        )


def test_lookup_maintain_rejects_missing_operator():
    with pytest.raises(RuntimeError, match="asu_hbm_index_maintain_aicpu"):
        DMPLookupMaintain(
            torch.device("cpu"),
            num_layers=1,
            max_microbatch_tokens=1,
            max_model_len=132000,
            custom_ops=SimpleNamespace(asu_hbm_index_lookup=lambda *args: None),
            attention_ops=FakeAttentionOps(),
            maintain_stream=object(),
        )
