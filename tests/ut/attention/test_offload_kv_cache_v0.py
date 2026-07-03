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

    def batch_put(self, cache_type, keys, values):
        for key, value in zip(keys, values, strict=True):
            self.store[(cache_type, key)] = value
        return True

    def batch_get(self, cache_type, keys):
        return [self.store.get((cache_type, key)) for key in keys]


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


def make_kv_cache():
    k_nope = torch.tensor(
        [[[[1.0, 1.5]], [[2.0, 2.5]], [[3.0, 3.5]], [[4.0, 4.5]]]],
        dtype=torch.float32,
    )
    k_pe = torch.tensor(
        [[[[10.0]], [[20.0]], [[30.0]], [[40.0]]]],
        dtype=torch.float32,
    )
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

    def test_prefill_persists_records_and_decode_loads_into_bypass_cache(self):
        client = FakeMicroKVClient()
        manager = OffloadKVCacheV0Manager(client=client, capacity=4, slot_table_size=16)
        kv_cache = make_kv_cache()
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
        self.assertEqual(stats.checked_items, 1)
        self.assertEqual(stats.loaded_items, 1)
        self.assertEqual(stats.mismatch_items, 0)
        self.assertEqual(manager.get_slot_id("req-a", 7, 2), 0)

    def test_microkv_miss_skips_validation_without_touching_original_flow(self):
        manager = OffloadKVCacheV0Manager(client=FakeMicroKVClient(), capacity=4, slot_table_size=16)
        metadata = SimpleMetadata(
            req_ids=["req-a"],
            token_req_indices_cpu=torch.tensor([0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([3], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([3], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            num_decode_tokens=1,
        )

        stats = manager.mock_lookup_and_validate(
            "model.layers.0.self_attn",
            make_kv_cache(),
            torch.tensor([[[1]]], dtype=torch.int32),
            metadata,
        )

        self.assertEqual(stats.checked_items, 0)
        self.assertEqual(stats.skipped_items, 1)
        self.assertEqual(stats.missing_items, 1)

    def test_eviction_invalidates_previous_token_slot(self):
        client = FakeMicroKVClient()
        manager = OffloadKVCacheV0Manager(client=client, capacity=1, slot_table_size=16)
        kv_cache = make_kv_cache()
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
            token_req_indices_cpu=torch.tensor([0, 0], dtype=torch.int32),
            token_positions_cpu=torch.tensor([2, 3], dtype=torch.int64),
            prefill_lens_cpu=torch.tensor([2], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            num_decode_tokens=2,
        )

        manager.mock_lookup_and_validate(
            "model.layers.0.self_attn",
            kv_cache,
            torch.tensor([[[0]], [[1]]], dtype=torch.int32),
            decode_metadata,
        )

        self.assertEqual(manager.get_slot_id("req-a", 0, 0), -1)
        self.assertEqual(manager.get_slot_id("req-a", 0, 1), 0)

    def test_cache_state_is_isolated_by_request_and_layer(self):
        client = FakeMicroKVClient()
        manager = OffloadKVCacheV0Manager(client=client, capacity=2, slot_table_size=16, strict=False)
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

    def test_mismatch_raises_with_strict_validation(self):
        client = FakeMicroKVClient()
        manager = OffloadKVCacheV0Manager(client=client, capacity=4, slot_table_size=16, strict=True)
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
            manager.mock_lookup_and_validate(
                "model.layers.0.self_attn",
                make_kv_cache(),
                torch.tensor([[[0]]], dtype=torch.int32),
                metadata,
            )


if __name__ == "__main__":
    unittest.main()
