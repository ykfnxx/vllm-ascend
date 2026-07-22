import argparse
import os
import sys
from pathlib import Path

import torch
import torch_npu


INDEX_CAPACITY = 144 * 1024
RESIDENT_SLOTS = 8 * 1024
QUERY_SLOTS = 2 * 1024
TOTAL_SLOTS = 10 * 1024
FIXED_MISS_COUNT = 300
FREE_HEAD_STRIDE = 16


def make_state(batch_size: int, device: str):
    token_to_slot = torch.full(
        (batch_size, INDEX_CAPACITY), -1, dtype=torch.int32, device=device
    )
    slot_to_token = torch.full(
        (batch_size, TOTAL_SLOTS), -1, dtype=torch.int32, device=device
    )
    initial = torch.arange(RESIDENT_SLOTS, dtype=torch.int32, device=device)
    token_to_slot[:, :RESIDENT_SLOTS] = initial
    slot_to_token[:, :RESIDENT_SLOTS] = initial
    free_slots = (
        torch.arange(RESIDENT_SLOTS, TOTAL_SLOTS, dtype=torch.int32, device=device)
        .view(1, -1)
        .expand(batch_size, -1)
        .clone()
    )
    free_head = torch.zeros(
        (batch_size, FREE_HEAD_STRIDE), dtype=torch.int32, device=device
    )
    pool_entries = torch.arange(batch_size, dtype=torch.int32, device=device)
    return token_to_slot, slot_to_token, free_slots, free_head, pool_entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "torch_extension"))
    os.environ.setdefault("DMP_LOOKUP_MAINTAIN_INSTALL_OPP_PATH", str(root / "opp"))
    import dmp_lookup_maintain_custom_ops  # noqa: F401

    state = make_state(args.batch_size, args.device)
    token_to_slot, slot_to_token, free_slots, free_head, pool_entries = state
    fixed_hit_count = QUERY_SLOTS - FIXED_MISS_COUNT
    hits = torch.arange(fixed_hit_count, dtype=torch.int32, device=args.device)
    misses = torch.arange(
        RESIDENT_SLOTS,
        RESIDENT_SLOTS + FIXED_MISS_COUNT,
        dtype=torch.int32,
        device=args.device,
    )
    query = torch.cat((hits, misses)).view(1, -1).expand(args.batch_size, -1).clone()

    seq_lens = torch.full(
        (args.batch_size,), 12000, dtype=torch.int32, device=args.device
    )
    needs_refill = torch.ones(args.batch_size, dtype=torch.bool, device=args.device)
    slot_out, miss_out, hit_sparse, miss_sparse, resident_token_ids = (
        torch.ops.dmp_lookup_maintain.asu_hbm_index_lookup(
            token_to_slot,
            slot_to_token,
            free_slots,
            free_head,
            pool_entries,
            query,
            seq_lens,
            needs_refill,
            args.batch_size,
        )
    )
    torch_npu.npu.synchronize()
    expected_misses = torch.zeros_like(query)
    expected_misses[:, fixed_hit_count:] = 1
    torch.testing.assert_close(miss_out.cpu(), expected_misses.cpu())
    expected_hit_slots = hits.view(1, -1).expand(args.batch_size, -1)
    torch.testing.assert_close(
        slot_out[:, :fixed_hit_count].cpu(), expected_hit_slots.cpu()
    )
    torch.testing.assert_close(
        hit_sparse[:, :fixed_hit_count].cpu(), expected_hit_slots.cpu()
    )
    assert torch.all(hit_sparse[:, fixed_hit_count:] == -1)
    torch.testing.assert_close(
        miss_sparse[:, fixed_hit_count:].cpu(),
        slot_out[:, fixed_hit_count:].cpu(),
    )

    block_size = 128
    blocks_per_full_row = (12000 + block_size - 1) // block_size
    full_blocks = args.batch_size * blocks_per_full_row
    full_block_table = torch.arange(
        full_blocks, dtype=torch.int32, device=args.device
    ).view(args.batch_size, blocks_per_full_row)
    cache_values = torch.arange(
        full_blocks * block_size,
        dtype=torch.float32,
        device=args.device,
    ).view(-1, 1)
    full_kv_cache = (
        cache_values.expand(-1, 16).to(torch.float16).view(full_blocks, block_size, 16)
    )
    full_k_rope = (
        (cache_values + 0.25)
        .expand(-1, 16)
        .to(torch.float16)
        .view(full_blocks, block_size, 16)
    )
    selection_blocks_per_row = TOTAL_SLOTS // block_size
    selection_blocks = args.batch_size * selection_blocks_per_row
    selection_block_table = torch.arange(
        selection_blocks, dtype=torch.int32, device=args.device
    ).view(args.batch_size, selection_blocks_per_row)
    selection_kv_cache = torch.zeros(
        (selection_blocks, block_size, 16), dtype=torch.float16, device=args.device
    )
    selection_k_rope = torch.zeros_like(selection_kv_cache)
    copied_count = torch.ops.dmp_lookup_maintain.dmp_lookup_kv_gather(
        selection_k_rope,
        selection_kv_cache,
        selection_block_table,
        resident_token_ids,
        query,
        slot_out,
        miss_out,
        needs_refill,
        full_k_rope,
        full_kv_cache,
        full_block_table,
        seq_lens,
    )
    torch_npu.npu.synchronize()
    actual_copied_count = copied_count.cpu().tolist()
    expected_copied_count = [FIXED_MISS_COUNT] * args.batch_size
    assert actual_copied_count == expected_copied_count, (
        f"KVGather copied_count mismatch: expected={expected_copied_count}, "
        f"actual={actual_copied_count}"
    )
    selection_flat = selection_kv_cache.view(args.batch_size, TOTAL_SLOTS, 16)
    assert selection_flat[0, RESIDENT_SLOTS, 0].cpu().item() == RESIDENT_SLOTS

    torch.ops.dmp_lookup_maintain.asu_hbm_index_maintain_aicpu(
        token_to_slot,
        slot_to_token,
        free_slots,
        free_head,
        pool_entries,
        slot_out,
        args.batch_size,
        7,
    )
    torch_npu.npu.synchronize()
    torch.testing.assert_close(
        free_head[:, 0].cpu(), torch.zeros(args.batch_size, dtype=torch.int32)
    )

    second_slots, second_misses, _, _, _ = (
        torch.ops.dmp_lookup_maintain.asu_hbm_index_lookup(
            token_to_slot,
            slot_to_token,
            free_slots,
            free_head,
            pool_entries,
            query,
            seq_lens,
            torch.zeros(args.batch_size, dtype=torch.bool, device=args.device),
            args.batch_size,
        )
    )
    torch_npu.npu.synchronize()
    torch.testing.assert_close(second_slots.cpu(), slot_out.cpu())
    torch.testing.assert_close(second_misses.cpu(), expected_misses.cpu())
    print(
        "Lookup/Maintain smoke OK: Lookup and Maintain keep a fixed workload "
        f"of {FIXED_MISS_COUNT} misses/evictions per request"
    )


if __name__ == "__main__":
    main()
