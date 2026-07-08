from dataclasses import dataclass
import unittest

import torch

from vllm_ascend.attention.offload_kv_cache_v0 import (
    OffloadKVCacheV0Manager,
    OffloadKVCacheV0MismatchError,
    MicroKVRecordError,
    pack_mla_token_record,
    parse_layer_id,
    unpack_mla_token_record,
)


class FakeMicroKVClient:

    def __init__(self):
        self.store = {}
        self.batch_get_keys = []

    def batch_put(self, cache_type, keys, values):
        for key, value in zip(keys, values, strict=True):
            self.store[(cache_type, key)] = value
        return True

    def batch_get(self, cache_type, keys):
        self.batch_get_keys.append((cache_type, list(keys)))
        return [self.store.get((cache_type, key)) for key in keys]


class FakeLookupOp:

    def __init__(self):
        self.calls = []

    def __call__(self, index, slot_to_index, free_slots, free_head, query_index, req_num):
        self.calls.append(
            {
                "index": index.clone(),
                "slot_to_index": slot_to_index.clone(),
                "free_slots": free_slots.clone(),
                "free_head": free_head.clone(),
                "query_index": query_index.clone(),
                "req_num": req_num,
            }
        )
        slot_out = torch.empty_like(query_index)
        for query_offset in range(query_index.shape[1]):
            token_pos = int(query_index[0, query_offset].item())
            slot_id = int(index[0, token_pos].item())
            if slot_id < 0:
                head = int(free_head[0].item())
                slot_id = int(free_slots[0, head].item())
                free_head[0] = head + 1
                index[0, token_pos] = slot_id
                slot_to_index[0, slot_id] = token_pos
            slot_out[0, query_offset] = slot_id
        return slot_out


class FakeMaintainOp:

    def __init__(self):
        self.calls = []

    def __call__(self, index, slot_to_index, free_slots, free_head, last_query_slots, req_num, seed):
        self.calls.append(
            {
                "index": index.clone(),
                "slot_to_index": slot_to_index.clone(),
                "free_slots": free_slots.clone(),
                "free_head": free_head.clone(),
                "last_query_slots": last_query_slots.clone(),
                "req_num": req_num,
                "seed": seed,
            }
        )
        free_head.zero_()


@dataclass
class SimpleMetadata:
    req_ids: list[str]
    token_req_indices_cpu: torch.Tensor
    token_positions_cpu: torch.Tensor
    prefill_lens_cpu: torch.Tensor
    block_table: torch.Tensor | None = None
    attn_state: str = "DecodeOnly"
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0


def make_kv_cache(token_count: int = 4):
    values = torch.arange(1, token_count + 1, dtype=torch.float32)
    k_nope = torch.stack((values, values + 0.5), dim=-1).view(1, token_count, 1, 2)
    k_pe = (values * 10).view(1, token_count, 1, 1)
    return k_nope, k_pe


class OffloadKVCacheV0Test(unittest.TestCase):

    def test_pack_unpack_roundtrip_preserves_tensor_metadata(self):
        k_nope = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
        k_pe = torch.tensor([[3.0]], dtype=torch.float16)

        record = pack_mla_token_record(k_nope, k_pe)
        got_nope, got_pe = unpack_mla_token_record(
            record,
            expected_k_nope_shape=(1, 2),
            expected_k_pe_shape=(1, 1),
            expected_dtype=torch.float16,
        )

        self.assertTrue(torch.equal(got_nope, k_nope))
        self.assertTrue(torch.equal(got_pe, k_pe))

    def test_unpack_rejects_dtype_mismatch(self):
        record = pack_mla_token_record(
            torch.tensor([[1.0]], dtype=torch.float16),
            torch.tensor([[2.0]], dtype=torch.float16),
        )

        with self.assertRaises(MicroKVRecordError):
            unpack_mla_token_record(record, expected_dtype=torch.float32)

    def test_parse_layer_id_uses_model_layer_number(self):
        self.assertEqual(parse_layer_id("model.layers.12.self_attn"), 12)

    def test_prefill_initializes_resident_window_and_real_lookup_validates_hit(self):
        client = FakeMicroKVClient()
        lookup_op = FakeLookupOp()
        maintain_op = FakeMaintainOp()
        manager = OffloadKVCacheV0Manager(
            client=client,
            lookup_op=lookup_op,
            maintain_op=maintain_op,
            index_size=16,
            slot_count=6,
            resident_slot_count=4,
            free_slot_count=2,
            query_count=4,
        )
        kv_cache = make_kv_cache(4)
        prefill_metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0, 0, 0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([0, 1, 2], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([3], dtype=torch.int32),
            attn_state="PrefillNoCache",
            num_actual_tokens=3,
        )

        put_stats = manager.persist_prefill_kv_to_microkv(
            "model.layers.7.self_attn",
            kv_cache,
            torch.tensor([0, 1, 2], dtype=torch.int64),
            prefill_metadata,
        )
        decode_metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([3], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([3], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            num_decode_tokens=1,
        )

        stats = manager.mock_lookup_and_validate(
            "model.layers.7.self_attn",
            kv_cache,
            torch.tensor([[[2]]], dtype=torch.int32),
            decode_metadata,
        )

        self.assertEqual(put_stats.written_items, 3)
        self.assertEqual(manager.get_slot_id("req-a", 7, 2), 2)
        self.assertEqual(stats.checked_items, 1)
        self.assertEqual(stats.mismatch_items, 0)
        self.assertEqual(len(lookup_op.calls), 1)
        self.assertEqual(lookup_op.calls[0]["req_num"], 1)
        self.assertTrue(torch.equal(lookup_op.calls[0]["query_index"], torch.tensor([[2, 2, 2, 2]], dtype=torch.int32)))
        self.assertEqual(len(maintain_op.calls), 0)

    def test_real_lookup_miss_loads_payload_and_calls_aicpu_maintain(self):
        client = FakeMicroKVClient()
        lookup_op = FakeLookupOp()
        maintain_op = FakeMaintainOp()
        manager = OffloadKVCacheV0Manager(
            client=client,
            lookup_op=lookup_op,
            maintain_op=maintain_op,
            index_size=16,
            slot_count=6,
            resident_slot_count=2,
            free_slot_count=4,
            query_count=4,
            maintain_seed=19,
        )
        kv_cache = make_kv_cache(4)
        prefill_metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0, 0, 0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([0, 1, 2], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([3], dtype=torch.int32),
            attn_state="PrefillNoCache",
            num_actual_tokens=3,
        )
        manager.persist_prefill_kv_to_microkv(
            "model.layers.0.self_attn",
            kv_cache,
            torch.tensor([0, 1, 2], dtype=torch.int64),
            prefill_metadata,
        )
        metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([3], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([3], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            num_decode_tokens=1,
        )

        stats = manager.validate_topk_with_real_hbm_index_ops(
            "model.layers.0.self_attn",
            kv_cache,
            torch.tensor([[[2]]], dtype=torch.int32),
            metadata,
        )

        self.assertEqual(stats.checked_items, 1)
        self.assertEqual(stats.loaded_items, 1)
        self.assertEqual(manager.get_slot_id("req-a", 0, 2), 2)
        self.assertEqual(len(maintain_op.calls), 1)
        self.assertEqual(maintain_op.calls[0]["req_num"], 1)
        self.assertEqual(maintain_op.calls[0]["seed"], 19)
        self.assertTrue(torch.equal(maintain_op.calls[0]["last_query_slots"], torch.tensor([[2, 2, 2, 2]], dtype=torch.int32)))

    def test_microkv_miss_raises_after_real_lookup(self):
        manager = OffloadKVCacheV0Manager(
            client=FakeMicroKVClient(),
            lookup_op=FakeLookupOp(),
            maintain_op=FakeMaintainOp(),
            index_size=16,
            slot_count=6,
            resident_slot_count=0,
            free_slot_count=6,
            query_count=4,
        )
        metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([3], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([3], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            num_decode_tokens=1,
        )

        with self.assertRaises(MicroKVRecordError):
            manager.validate_topk_with_real_hbm_index_ops(
                "model.layers.0.self_attn",
                make_kv_cache(),
                torch.tensor([[[1]]], dtype=torch.int32),
                metadata,
            )

    def test_cache_state_is_isolated_by_request_and_layer(self):
        client = FakeMicroKVClient()
        manager = OffloadKVCacheV0Manager(
            client=client,
            lookup_op=FakeLookupOp(),
            maintain_op=FakeMaintainOp(),
            index_size=16,
            slot_count=6,
            resident_slot_count=2,
            free_slot_count=4,
            query_count=4,
            strict=False,
        )
        kv_cache = make_kv_cache()
        prefill_metadata = SimpleMetadata(
            req_ids=["req-a", "req-b"],
            token_req_indices_cpu=torch.tensor([0, 1], dtype=torch.int32),
            token_positions_cpu=torch.tensor([0, 0], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
            attn_state="PrefillNoCache",
            num_actual_tokens=2,
        )
        manager.persist_prefill_kv_to_microkv(
            "model.layers.0.self_attn",
            kv_cache,
            torch.tensor([0, 1], dtype=torch.int64),
            prefill_metadata,
        )

        for req_id in ["req-a", "req-b"]:
            metadata = SimpleMetadata(
                req_ids=["req-a", "req-b"],
                token_req_indices_cpu=torch.tensor([0 if req_id == "req-a" else 1], dtype=torch.int32),
                token_positions_cpu=torch.tensor([1], dtype=torch.int64),
                prefill_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
                block_table=torch.tensor([[0], [0]], dtype=torch.int32),
                num_decode_tokens=1,
            )
            manager.mock_lookup_and_validate(
                "model.layers.0.self_attn",
                kv_cache,
                torch.tensor([[[0]]], dtype=torch.int32),
                metadata,
            )

        self.assertEqual(manager.get_slot_id("req-a", 0, 0), 0)
        self.assertEqual(manager.get_slot_id("req-b", 0, 0), 0)
        self.assertEqual(manager.get_slot_id("req-a", 1, 0), -1)

    def test_query_index_filters_deduplicates_and_pads_before_lookup(self):
        client = FakeMicroKVClient()
        lookup_op = FakeLookupOp()
        manager = OffloadKVCacheV0Manager(
            client=client,
            lookup_op=lookup_op,
            maintain_op=FakeMaintainOp(),
            index_size=16,
            slot_count=6,
            resident_slot_count=3,
            free_slot_count=3,
            query_count=4,
        )
        kv_cache = make_kv_cache(4)
        prefill_metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0, 0, 0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([0, 1, 2], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([3], dtype=torch.int32),
            attn_state="PrefillNoCache",
            num_actual_tokens=3,
        )
        manager.persist_prefill_kv_to_microkv(
            "model.layers.0.self_attn",
            kv_cache,
            torch.tensor([0, 1, 2], dtype=torch.int64),
            prefill_metadata,
        )
        decode_metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([3], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([3], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            num_decode_tokens=1,
        )

        stats = manager.validate_topk_with_real_hbm_index_ops(
            "model.layers.0.self_attn",
            kv_cache,
            torch.tensor([[[1, -1, 4, 1, 2]]], dtype=torch.int32),
            decode_metadata,
        )

        self.assertEqual(stats.checked_items, 3)
        self.assertTrue(torch.equal(lookup_op.calls[0]["query_index"], torch.tensor([[1, 2, 1, 1]], dtype=torch.int32)))

    def test_unique_query_overflow_raises_before_lookup(self):
        lookup_op = FakeLookupOp()
        manager = OffloadKVCacheV0Manager(
            client=FakeMicroKVClient(),
            lookup_op=lookup_op,
            maintain_op=FakeMaintainOp(),
            index_size=16,
            slot_count=6,
            resident_slot_count=3,
            free_slot_count=3,
            query_count=2,
        )
        metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([3], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([3], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            num_decode_tokens=1,
        )

        with self.assertRaises(ValueError):
            manager.validate_topk_with_real_hbm_index_ops(
                "model.layers.0.self_attn",
                make_kv_cache(),
                torch.tensor([[[0, 1, 2]]], dtype=torch.int32),
                metadata,
            )
        self.assertEqual(len(lookup_op.calls), 0)

    def test_compact_materializes_generated_tokens_from_original_kv_cache(self):
        client = FakeMicroKVClient()
        lookup_op = FakeLookupOp()
        maintain_op = FakeMaintainOp()
        manager = OffloadKVCacheV0Manager(
            client=client,
            lookup_op=lookup_op,
            maintain_op=maintain_op,
            index_size=16,
            slot_count=4,
            resident_slot_count=2,
            free_slot_count=2,
            query_count=4,
            compact_sfa_enabled=True,
            max_pinned_reqs=1,
            block_size=2,
        )
        k_nope = torch.arange(8, dtype=torch.float32).view(4, 2, 1, 1)
        k_pe = (torch.arange(8, dtype=torch.float32) + 100).view(4, 2, 1, 1)
        kv_cache = (k_nope.clone(), k_pe.clone())
        prefill_metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0, 0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([0, 1], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([2], dtype=torch.int32),
            attn_state="PrefillNoCache",
            num_actual_tokens=2,
        )
        manager.persist_prefill_kv_to_microkv(
            "model.layers.0.self_attn",
            kv_cache,
            torch.tensor([0, 1], dtype=torch.int64),
            prefill_metadata,
        )
        decode_metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([3], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([2], dtype=torch.int32),
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            num_decode_tokens=1,
        )

        compact_inputs = manager.prepare_compact_sfa_inputs(
            "model.layers.0.self_attn",
            kv_cache,
            torch.tensor([[[1, 3]]], dtype=torch.int32),
            decode_metadata,
            torch.tensor([4], dtype=torch.int32),
        )

        self.assertTrue(torch.equal(compact_inputs.topk_indices, torch.tensor([[[1, 2]]], dtype=torch.int32)))
        self.assertTrue(torch.equal(compact_inputs.block_table, torch.tensor([[2, 3]], dtype=torch.int32)))
        self.assertEqual(len(client.batch_get_keys), 1)
        self.assertEqual(client.batch_get_keys[0][1], [manager.make_key("req-a", 0, 1)])
        self.assertTrue(torch.equal(kv_cache[0].view(-1, 1, 1)[6], torch.tensor([[3.0]])))
        self.assertTrue(torch.equal(kv_cache[1].view(-1, 1, 1)[6], torch.tensor([[103.0]])))

    def test_mismatch_raises_with_strict_validation(self):
        client = FakeMicroKVClient()
        manager = OffloadKVCacheV0Manager(
            client=client,
            lookup_op=FakeLookupOp(),
            maintain_op=FakeMaintainOp(),
            index_size=16,
            slot_count=6,
            resident_slot_count=0,
            free_slot_count=6,
            query_count=4,
            strict=True,
        )
        wrong_record = pack_mla_token_record(
            torch.tensor([[99.0, 99.5]], dtype=torch.float32),
            torch.tensor([[99.0]], dtype=torch.float32),
        )
        key = manager.make_key("req-a", 0, 0)
        client.batch_put(manager.cache_type, [key], [wrong_record])
        metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([1], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([1], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            num_decode_tokens=1,
        )

        with self.assertRaises(OffloadKVCacheV0MismatchError):
            manager.validate_topk_with_real_hbm_index_ops(
                "model.layers.0.self_attn",
                make_kv_cache(),
                torch.tensor([[[0]]], dtype=torch.int32),
                metadata,
            )


if __name__ == "__main__":
    unittest.main()
