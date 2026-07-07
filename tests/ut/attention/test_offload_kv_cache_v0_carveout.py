import unittest

from vllm_ascend.attention.offload_kv_cache_v0_ownership import (
    BlockOwnershipRegistry,
    build_static_offload_blocks,
    compact_blocks_per_req,
    inflated_tensor_size,
    offload_reserved_blocks,
    offload_reserved_bytes,
)


class OffloadReservedBlocksTest(unittest.TestCase):

    def test_reserved_blocks_is_product(self):
        self.assertEqual(offload_reserved_blocks(max_pinned_reqs=3, blocks_per_req=80), 240)

    def test_reserved_blocks_matches_design_example(self):
        # SLOT_COUNT = 10240, block_size = 128 -> 80 compact blocks per request.
        blocks_per_req = compact_blocks_per_req(slot_count=10240, block_size=128)
        self.assertEqual(blocks_per_req, 80)
        self.assertEqual(offload_reserved_blocks(max_pinned_reqs=1, blocks_per_req=blocks_per_req), 80)

    def test_reserved_blocks_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            offload_reserved_blocks(max_pinned_reqs=0, blocks_per_req=80)
        with self.assertRaises(ValueError):
            offload_reserved_blocks(max_pinned_reqs=3, blocks_per_req=0)


class OffloadReservedBytesTest(unittest.TestCase):

    def test_reserved_bytes_is_product(self):
        self.assertEqual(offload_reserved_bytes(reserved_blocks=80, page_size_bytes_total=1024), 80 * 1024)

    def test_reserved_bytes_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            offload_reserved_bytes(reserved_blocks=0, page_size_bytes_total=1024)
        with self.assertRaises(ValueError):
            offload_reserved_bytes(reserved_blocks=80, page_size_bytes_total=0)


class InflatedTensorSizeTest(unittest.TestCase):

    def test_inflate_adds_reserved_blocks_worth_of_bytes(self):
        # 100 scheduler blocks of 512 bytes each, reserve 4 blocks.
        size = 100 * 512
        self.assertEqual(
            inflated_tensor_size(size_bytes=size, page_size_bytes=512, reserved_blocks=4),
            104 * 512,
        )

    def test_inflate_rejects_non_multiple_size(self):
        with self.assertRaises(ValueError):
            inflated_tensor_size(size_bytes=513, page_size_bytes=512, reserved_blocks=4)

    def test_inflate_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            inflated_tensor_size(size_bytes=0, page_size_bytes=512, reserved_blocks=4)
        with self.assertRaises(ValueError):
            inflated_tensor_size(size_bytes=512, page_size_bytes=0, reserved_blocks=4)


class StaticCarveoutInvariantTest(unittest.TestCase):
    """End-to-end check of the allocator carve-out arithmetic.

    Simulates the worker-side flow without torch/vLLM:
      1. Engine sizes the KV tensor to ``scheduler_blocks`` (block ids [0, N) that the
         normal allocator may hand out).
      2. Worker inflates the physical tensor by ``reserved_blocks`` via
         ``inflated_tensor_size``.
      3. The offload pool is the tail of the inflated tensor and must start exactly at
         ``scheduler_blocks`` so it is disjoint from the scheduler's block range.
    """

    def _carve(self, scheduler_blocks, page_size_bytes, max_pinned_reqs, blocks_per_req):
        reserved_blocks = offload_reserved_blocks(max_pinned_reqs, blocks_per_req)
        scheduler_size = scheduler_blocks * page_size_bytes
        inflated = inflated_tensor_size(scheduler_size, page_size_bytes, reserved_blocks)
        total_blocks = inflated // page_size_bytes
        offload_blocks = build_static_offload_blocks(total_blocks, max_pinned_reqs, blocks_per_req)
        return reserved_blocks, total_blocks, offload_blocks

    def test_offload_tail_starts_exactly_at_scheduler_block_count(self):
        scheduler_blocks = 100
        reserved_blocks, total_blocks, offload_blocks = self._carve(
            scheduler_blocks=scheduler_blocks,
            page_size_bytes=512,
            max_pinned_reqs=2,
            blocks_per_req=4,
        )
        self.assertEqual(reserved_blocks, 8)
        self.assertEqual(total_blocks, scheduler_blocks + reserved_blocks)
        # Offload pool is the tail; its first id equals the scheduler-visible block count.
        self.assertEqual(offload_blocks, list(range(scheduler_blocks, total_blocks)))
        self.assertEqual(offload_blocks[0], scheduler_blocks)

    def test_scheduler_range_and_offload_range_are_disjoint_and_cover_all(self):
        scheduler_blocks = 64
        reserved_blocks, total_blocks, offload_blocks = self._carve(
            scheduler_blocks=scheduler_blocks,
            page_size_bytes=256,
            max_pinned_reqs=1,
            blocks_per_req=80,
        )
        scheduler_range = set(range(scheduler_blocks))
        offload_range = set(offload_blocks)
        self.assertEqual(scheduler_range & offload_range, set())
        self.assertEqual(scheduler_range | offload_range, set(range(total_blocks)))

    def test_registry_after_carveout_matches_scheduler_split(self):
        scheduler_blocks = 32
        _, total_blocks, offload_blocks = self._carve(
            scheduler_blocks=scheduler_blocks,
            page_size_bytes=128,
            max_pinned_reqs=1,
            blocks_per_req=8,
        )
        registry = BlockOwnershipRegistry(total_blocks=total_blocks, offload_blocks=offload_blocks)

        # Every block the scheduler can hand out is a NORMAL_KV_BLOCK.
        self.assertEqual(registry.normal_kv_blocks(), list(range(scheduler_blocks)))
        # Original K/V metadata restricted to scheduler blocks passes; offload blocks are rejected.
        registry.assert_original_kv_blocks(range(scheduler_blocks))
        for offload_block in offload_blocks:
            with self.assertRaises(ValueError):
                registry.assert_original_kv_blocks([offload_block])
        # Compact metadata is exactly the offload blocks.
        registry.assert_compact_kv_blocks(offload_blocks)

    def test_multi_layer_reserved_bytes_equals_total_inflation(self):
        # Reserving R blocks across L layers of equal page size must cost exactly
        # R * sum(page_size per layer), which is what the worker subtracts in
        # determine_available_memory before inflating each layer tensor.
        reserved_blocks = 8
        per_layer_page_size = 512
        num_layers = 3
        page_size_total = per_layer_page_size * num_layers
        reserved_bytes = offload_reserved_bytes(reserved_blocks, page_size_total)

        per_layer_added = 0
        for _ in range(num_layers):
            base = 100 * per_layer_page_size
            per_layer_added += inflated_tensor_size(base, per_layer_page_size, reserved_blocks) - base
        self.assertEqual(reserved_bytes, per_layer_added)


class StaticCarveoutFailFastTest(unittest.TestCase):

    def test_reserved_pool_must_leave_normal_blocks(self):
        # reserved == total -> no normal blocks left -> fail fast (design section 11).
        with self.assertRaises(ValueError):
            build_static_offload_blocks(total_blocks=8, max_pinned_reqs=2, blocks_per_req=4)

    def test_reserved_pool_smaller_than_total_is_ok(self):
        blocks = build_static_offload_blocks(total_blocks=9, max_pinned_reqs=2, blocks_per_req=4)
        self.assertEqual(blocks, [1, 2, 3, 4, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
