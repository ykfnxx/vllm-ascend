import unittest

from vllm_ascend.attention.offload_kv_cache_v0_ownership import (
    INDEXER_BLOCK,
    INDEXER_KEY_DOMAIN,
    KV_PAYLOAD_DOMAIN,
    NORMAL_KV_BLOCK,
    OFFLOAD_KV_BLOCK,
    BlockOwnershipRegistry,
    build_compact_block_table_row,
    compact_blocks_per_req,
    physical_slot_for_compact_slot,
)


class OffloadKVCacheV0OwnershipTest(unittest.TestCase):

    def test_static_carveout_marks_only_kv_payload_offload_blocks(self):
        registry = BlockOwnershipRegistry(
            total_blocks=12,
            offload_blocks=[8, 9, 10, 11],
        )

        self.assertEqual(registry.owner(KV_PAYLOAD_DOMAIN, 7), NORMAL_KV_BLOCK)
        self.assertEqual(registry.owner(KV_PAYLOAD_DOMAIN, 8), OFFLOAD_KV_BLOCK)
        self.assertEqual(registry.owner(INDEXER_KEY_DOMAIN, 8), INDEXER_BLOCK)
        self.assertEqual(registry.normal_kv_blocks(), list(range(8)))

    def test_original_kv_blocks_reject_offload_owned_blocks(self):
        registry = BlockOwnershipRegistry(
            total_blocks=8,
            offload_blocks=[6, 7],
        )

        registry.assert_original_kv_blocks([0, 1, 5])
        with self.assertRaises(ValueError):
            registry.assert_original_kv_blocks([0, 6])

    def test_compact_block_table_accepts_only_offload_owned_blocks(self):
        registry = BlockOwnershipRegistry(
            total_blocks=16,
            offload_blocks=[12, 13, 14, 15],
        )

        self.assertEqual(build_compact_block_table_row(registry, [12, 13]), [12, 13])
        with self.assertRaises(ValueError):
            build_compact_block_table_row(registry, [11, 12])

    def test_compact_slot_maps_through_compact_block_table(self):
        self.assertEqual(compact_blocks_per_req(slot_count=10, block_size=4), 3)
        self.assertEqual(
            physical_slot_for_compact_slot(
                slot_id=6,
                block_size=4,
                compact_block_table_row=[20, 21, 22],
            ),
            21 * 4 + 2,
        )

        with self.assertRaises(ValueError):
            physical_slot_for_compact_slot(
                slot_id=12,
                block_size=4,
                compact_block_table_row=[20, 21, 22],
            )


if __name__ == "__main__":
    unittest.main()
