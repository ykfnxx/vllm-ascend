import unittest

from vllm_ascend.attention.offload_kv_cache_v0_ref_ops import (
    NOT_FOUND,
    hash32,
    lookup_request,
    maintain_request,
)


def make_state(index_size, slot_count, resident_slot_count, free_slot_count):
    index = [NOT_FOUND] * index_size
    slot_to_index = [NOT_FOUND] * slot_count
    free_slots = list(range(resident_slot_count, resident_slot_count + free_slot_count))
    return index, slot_to_index, free_slots


def set_resident(index, slot_to_index, token_pos, slot_id):
    index[token_pos] = slot_id
    slot_to_index[slot_id] = token_pos


class Hash32Test(unittest.TestCase):

    def test_hash_of_zero_is_zero(self):
        self.assertEqual(hash32(0), 0)

    def test_hash_is_deterministic_and_32bit(self):
        for x in [1, 7, 123456, 0xDEADBEEF, 0xFFFFFFFF]:
            h = hash32(x)
            self.assertEqual(h, hash32(x))
            self.assertTrue(0 <= h <= 0xFFFFFFFF)

    def test_hash_distinguishes_inputs(self):
        self.assertNotEqual(hash32(1), hash32(2))


class LookupRequestTest(unittest.TestCase):

    def setUp(self):
        self.index, self.slot_to_index, self.free_slots = make_state(
            index_size=16, slot_count=8, resident_slot_count=4, free_slot_count=4
        )
        # Residents: token 0 -> slot 0, token 1 -> slot 1.
        set_resident(self.index, self.slot_to_index, 0, 0)
        set_resident(self.index, self.slot_to_index, 1, 1)

    def test_hit_returns_existing_slot_without_allocating(self):
        slot_out, head = lookup_request(self.index, self.slot_to_index, self.free_slots, 0, [0, 1])
        self.assertEqual(slot_out, [0, 1])
        self.assertEqual(head, 0)  # nothing allocated
        self.assertEqual(self.free_slots, [4, 5, 6, 7])  # untouched

    def test_miss_allocates_from_free_pool_and_wires_maps(self):
        slot_out, head = lookup_request(self.index, self.slot_to_index, self.free_slots, 0, [5])
        self.assertEqual(slot_out, [4])  # free_slots[0]
        self.assertEqual(head, 1)
        self.assertEqual(self.index[5], 4)
        self.assertEqual(self.slot_to_index[4], 5)

    def test_duplicate_token_consumes_single_slot(self):
        slot_out, head = lookup_request(self.index, self.slot_to_index, self.free_slots, 0, [6, 6, 6])
        self.assertEqual(slot_out, [4, 4, 4])
        self.assertEqual(head, 1)
        self.assertEqual(self.index[6], 4)

    def test_mixed_resident_and_misses(self):
        slot_out, head = lookup_request(self.index, self.slot_to_index, self.free_slots, 0, [0, 7, 7, 1, 8])
        # 0,1 resident; 7 allocates slot 4 (reused for the duplicate); 8 allocates slot 5.
        self.assertEqual(slot_out, [0, 4, 4, 1, 5])
        self.assertEqual(head, 2)
        self.assertEqual(self.index[7], 4)
        self.assertEqual(self.index[8], 5)

    def test_allocation_continues_from_given_free_head(self):
        slot_out, head = lookup_request(self.index, self.slot_to_index, self.free_slots, 2, [9])
        self.assertEqual(slot_out, [6])  # free_slots[2]
        self.assertEqual(head, 3)


class MaintainRequestTest(unittest.TestCase):

    def _occupied_slots(self, slot_to_index):
        return {s for s, tok in enumerate(slot_to_index) if tok != NOT_FOUND}

    def _assert_bijection(self, index, slot_to_index):
        for slot, token in enumerate(slot_to_index):
            if token != NOT_FOUND:
                self.assertEqual(index[token], slot, f"slot_to_index[{slot}]={token} but index[{token}]={index[token]}")

    def test_zero_free_head_is_noop(self):
        index, slot_to_index, free_slots = make_state(16, 8, 4, 4)
        set_resident(index, slot_to_index, 0, 0)
        before = (list(index), list(slot_to_index), list(free_slots))
        head = maintain_request(index, slot_to_index, free_slots, 0, [0] * 4, seed=0, req_id=0, slot_count=8)
        self.assertEqual(head, 0)
        self.assertEqual((index, slot_to_index, free_slots), before)

    def test_reclaims_exactly_free_head_unprotected_slots(self):
        index, slot_to_index, free_slots = make_state(16, 8, 0, 8)
        # Occupy 6 slots: tokens 10..15 -> slots 0..5.
        for slot in range(6):
            set_resident(index, slot_to_index, 10 + slot, slot)
        occupied_before = self._occupied_slots(slot_to_index)
        protected = [4, 4, 4, 4]  # protect slot 4
        free_head = 2

        head = maintain_request(index, slot_to_index, free_slots, free_head, protected, seed=1, req_id=0, slot_count=8)

        self.assertEqual(head, 0)
        occupied_after = self._occupied_slots(slot_to_index)
        evicted = occupied_before - occupied_after
        self.assertEqual(len(evicted), free_head)  # exactly free_head slots reclaimed
        self.assertNotIn(4, evicted)  # protected slot survives
        self.assertIn(4, occupied_after)
        # Reclaimed slots pushed to the bottom of the free pool.
        self.assertEqual(set(free_slots[:free_head]), evicted)
        # Evicted tokens' forward map cleared.
        for slot in evicted:
            self.assertNotIn(slot, self._occupied_slots(slot_to_index))
        self._assert_bijection(index, slot_to_index)

    def test_protected_slots_are_never_evicted(self):
        index, slot_to_index, free_slots = make_state(16, 8, 0, 8)
        for slot in range(5):
            set_resident(index, slot_to_index, 10 + slot, slot)
        protected = [0, 1, 2]  # protect slots 0,1,2
        head = maintain_request(index, slot_to_index, free_slots, 2, protected, seed=3, req_id=0, slot_count=8)
        self.assertEqual(head, 0)
        for slot in (0, 1, 2):
            self.assertNotEqual(slot_to_index[slot], NOT_FOUND)

    def test_raises_when_not_enough_reclaimable_slots(self):
        index, slot_to_index, free_slots = make_state(16, 8, 0, 8)
        # Only 2 occupied slots and both protected -> cannot reclaim 2.
        set_resident(index, slot_to_index, 10, 0)
        set_resident(index, slot_to_index, 11, 1)
        with self.assertRaises(ValueError):
            maintain_request(index, slot_to_index, free_slots, 2, [0, 1], seed=0, req_id=0, slot_count=8)


class LookupMaintainCycleTest(unittest.TestCase):

    def test_bidirectional_map_stays_consistent_across_cycle(self):
        index, slot_to_index, free_slots = make_state(index_size=32, slot_count=8, resident_slot_count=4, free_slot_count=4)
        # Prefill residents: tokens 0..3 -> slots 0..3.
        for tok in range(4):
            set_resident(index, slot_to_index, tok, tok)

        # Decode step 1: query residents + two new tokens (misses).
        slot_out, head = lookup_request(index, slot_to_index, free_slots, 0, [0, 1, 20, 21])
        self.assertEqual(head, 2)
        for slot, token in enumerate(slot_to_index):
            if token != NOT_FOUND:
                self.assertEqual(index[token], slot)

        # maintain reclaims the two freshly allocated slots (protect only the residents).
        maintain_request(index, slot_to_index, free_slots, head, [0, 1, 2, 3], seed=7, req_id=0, slot_count=8)

        # Decode step 2: new tokens must be servable again from the refilled pool.
        slot_out2, head2 = lookup_request(index, slot_to_index, free_slots, 0, [22, 23])
        self.assertEqual(head2, 2)
        self.assertEqual(len(set(slot_out2)), 2)  # two distinct fresh slots
        for slot, token in enumerate(slot_to_index):
            if token != NOT_FOUND:
                self.assertEqual(index[token], slot)


if __name__ == "__main__":
    unittest.main()
