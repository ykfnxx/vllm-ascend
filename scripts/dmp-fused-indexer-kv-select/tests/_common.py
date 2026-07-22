#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

torch = None
torch_npu = None


SPARSE_COUNT = 2048
CACHE_SLOTS_CAPACITY = 262144
MAX_SLOT_ID_EXCLUSIVE = 16383


def import_torch_npu():
    global torch
    global torch_npu
    import torch as torch_module
    import torch_npu as torch_npu_module

    torch = torch_module
    torch_npu = torch_npu_module

    repo_root = Path(__file__).resolve().parents[1]
    extension_dir = repo_root / "torch_extension"
    if extension_dir.exists():
        sys.path.insert(0, str(extension_dir))
    import lightning_indexer_decode_custom_ops  # noqa: F401


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bs", type=int, default=24)
    parser.add_argument("--min-seqlen", type=int, default=32768)
    parser.add_argument("--max-seqlen", type=int, default=65536)
    parser.add_argument("--q-heads", type=int, default=64)
    parser.add_argument("--k-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--block-base", type=int, default=1)
    parser.add_argument("--max-blocks-per-batch", type=int, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cache-size", type=int, default=8192)
    parser.add_argument("--min-miss-count", type=int, default=0)
    parser.add_argument("--max-miss-count", type=int, default=200)
    return parser.parse_args()


def validate_args(args):
    if args.bs <= 0:
        raise ValueError("bs must be positive.")
    if args.min_seqlen < SPARSE_COUNT:
        raise ValueError("min-seqlen must be >= 2048.")
    if args.max_seqlen > CACHE_SLOTS_CAPACITY:
        raise ValueError("max-seqlen must be <= 262144.")
    if args.min_seqlen > args.max_seqlen:
        raise ValueError("min-seqlen must be <= max-seqlen.")
    seqlen_value_count = args.max_seqlen - args.min_seqlen + 1
    if args.min_seqlen != args.max_seqlen and seqlen_value_count < args.bs:
        raise ValueError("The seqlen range must contain at least bs distinct values.")
    if args.q_heads != 64:
        raise ValueError("q-heads is fixed to 64.")
    if args.k_heads != 1:
        raise ValueError("k-heads is fixed to 1.")
    if args.head_dim != 128:
        raise ValueError("head-dim is fixed to 128.")
    if args.block_size <= 0 or args.block_size % 16 != 0 or args.block_size > 1024:
        raise ValueError("block-size must be a multiple of 16 in (0, 1024].")
    if args.block_base < 0:
        raise ValueError("block-base must be >= 0.")
    if args.min_miss_count < 0 or args.max_miss_count > SPARSE_COUNT:
        raise ValueError("0 <= min-miss-count <= max-miss-count <= 2048 is required.")
    if args.min_miss_count > args.max_miss_count:
        raise ValueError("min-miss-count must be <= max-miss-count.")
    if args.cache_size < SPARSE_COUNT:
        raise ValueError("cache-size must be >= 2048.")
    if args.cache_size > MAX_SLOT_ID_EXCLUSIVE:
        raise ValueError("cache-size must be <= 16383 because slot14 value 0x3fff is reserved for invalid.")
    if args.cache_size > args.min_seqlen:
        raise ValueError("cache-size must be <= min-seqlen.")
    if args.cache_size + args.max_miss_count > args.min_seqlen:
        raise ValueError(
            "cache-size + max-miss-count must be <= min-seqlen so every request has enough candidates."
        )
    active_blocks_per_batch = (args.max_seqlen + args.block_size - 1) // args.block_size
    max_blocks_per_batch = active_blocks_per_batch if args.max_blocks_per_batch is None else args.max_blocks_per_batch
    if max_blocks_per_batch < active_blocks_per_batch:
        raise ValueError("max-blocks-per-batch must be >= ceil(max-seqlen / block-size).")
    if max_blocks_per_batch * args.block_size > CACHE_SLOTS_CAPACITY:
        raise ValueError("max-blocks-per-batch * block-size must be <= 262144.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be >= 0 and iters must be > 0.")


def dtype_from_name(name):
    return torch.bfloat16 if name == "bf16" else torch.float16


def sample_seqlens(args):
    if args.min_seqlen == args.max_seqlen:
        return [args.min_seqlen] * args.bs
    return random.Random(args.seed).sample(
        range(args.min_seqlen, args.max_seqlen + 1),
        args.bs,
    )


def run_decode(tensors):
    custom_ops = getattr(torch.ops, "custom", None)
    custom_op = getattr(custom_ops, "npu_lightning_indexer_decode", None) if custom_ops is not None else None
    if custom_op is not None:
        output = custom_op(
            tensors["query"],
            tensors["key"],
            tensors["weights"],
            tensors["actual_seq_lengths_key"],
            tensors["block_table"],
        )
    else:
        output = torch_npu.npu_lightning_indexer_decode(
            tensors["query"],
            tensors["key"],
            tensors["weights"],
            actual_seq_lengths_key=tensors["actual_seq_lengths_key"],
            block_table=tensors["block_table"],
        )
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f"Expected one tensor output from npu_lightning_indexer_decode, got {output!r}.")
    return output


def run_decode_update(tensors):
    custom_ops = getattr(torch.ops, "custom", None)
    custom_op = getattr(custom_ops, "npu_lightning_indexer_decode_update", None) if custom_ops is not None else None
    if custom_op is not None:
        output = custom_op(
            tensors["query"],
            tensors["key"],
            tensors["weights"],
            tensors["cache_slots"],
            tensors["actual_seq_lengths_key"],
            tensors["block_table"],
        )
    else:
        output = torch_npu.npu_lightning_indexer_decode_update(
            tensors["query"],
            tensors["key"],
            tensors["weights"],
            tensors["cache_slots"],
            actual_seq_lengths_key=tensors["actual_seq_lengths_key"],
            block_table=tensors["block_table"],
        )
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise RuntimeError(f"Expected three outputs from npu_lightning_indexer_decode_update, got {output!r}.")
    return output[0], output[1], output[2]


def allocate_inputs(args, device):
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    dtype = dtype_from_name(args.dtype)
    seqlens = sample_seqlens(args)
    active_blocks_per_batch = [
        (seqlen + args.block_size - 1) // args.block_size for seqlen in seqlens
    ]
    max_active_blocks_per_batch = (args.max_seqlen + args.block_size - 1) // args.block_size
    max_blocks_per_batch = (
        max_active_blocks_per_batch if args.max_blocks_per_batch is None else args.max_blocks_per_batch
    )
    used_blocks = args.bs * max_blocks_per_batch
    num_blocks = args.block_base + used_blocks

    query = torch.empty((args.bs, args.q_heads, args.head_dim), dtype=dtype, device=device)
    key = torch.empty((num_blocks, args.block_size, args.k_heads, args.head_dim), dtype=dtype, device=device)
    weights = torch.empty((args.bs, args.q_heads), dtype=dtype, device=device)
    query.uniform_(-1.0, 1.0)
    key.uniform_(-1.0, 1.0)
    weights.uniform_(-1.0, 1.0)

    actual_seq_lengths_key = torch.tensor(seqlens, dtype=torch.int32, device=device)
    block_table = torch.arange(
        args.block_base,
        args.block_base + used_blocks,
        dtype=torch.int32,
        device=device,
    ).reshape(args.bs, max_blocks_per_batch)
    cache_slots = torch.full((args.bs, CACHE_SLOTS_CAPACITY), -1, dtype=torch.int32, device=device)

    return {
        "query": query,
        "key": key,
        "weights": weights,
        "cache_slots": cache_slots,
        "actual_seq_lengths_key": actual_seq_lengths_key,
        "block_table": block_table,
        "seqlens": seqlens,
        "active_blocks_per_batch": active_blocks_per_batch,
        "max_blocks_per_batch": max_blocks_per_batch,
        "num_blocks": num_blocks,
    }


def sample_tensor_values(pool, count, generator):
    if count < 0:
        raise ValueError("sample count must be non-negative.")
    if count == 0:
        return torch.empty((0,), dtype=torch.long)
    if pool.numel() < count:
        raise ValueError(f"Cannot sample {count} values from pool of {pool.numel()}.")
    order = torch.randperm(pool.numel(), generator=generator)[:count]
    return pool[order].to(torch.long)


def build_cache_slots(args, reference_cpu, seqlens):
    generator = torch.Generator()
    generator.manual_seed(args.seed + 1)
    target_miss_count = torch.randint(
        args.min_miss_count,
        args.max_miss_count + 1,
        (args.bs,),
        generator=generator,
        dtype=torch.int32,
    )
    cache_slots_cpu = torch.full((args.bs, CACHE_SLOTS_CAPACITY), -1, dtype=torch.int32)

    for batch_idx in range(args.bs):
        seqlen = seqlens[batch_idx]
        miss_count = int(target_miss_count[batch_idx].item())
        hit_count = SPARSE_COUNT - miss_count
        other_count = args.cache_size - hit_count

        topk = reference_cpu[batch_idx, 0].to(torch.long)
        hit_index = sample_tensor_values(topk, hit_count, generator)

        outside_mask = torch.ones(seqlen, dtype=torch.bool)
        outside_mask[topk] = False
        outside_index = torch.arange(seqlen, dtype=torch.long)[outside_mask]
        other_index = sample_tensor_values(outside_index, other_count, generator)

        cached_index = torch.cat((hit_index, other_index), dim=0)
        shuffled_slots = torch.randperm(args.cache_size, generator=generator, dtype=torch.int32)
        cache_slots_cpu[batch_idx, cached_index] = shuffled_slots

    return cache_slots_cpu, target_miss_count


def check_cache_exact(args, cache_cpu, seqlens, label):
    expected_slots = torch.arange(args.cache_size, dtype=torch.int32)
    for batch_idx in range(args.bs):
        seqlen = seqlens[batch_idx]
        row = cache_cpu[batch_idx]
        valid_pos = (row >= 0).nonzero(as_tuple=False).flatten()
        if valid_pos.numel() != args.cache_size:
            raise AssertionError(
                f"{label} batch {batch_idx} has {valid_pos.numel()} valid slots, expected {args.cache_size}."
            )
        if bool((valid_pos >= seqlen).any().item()):
            pos = int(valid_pos[(valid_pos >= seqlen).nonzero(as_tuple=False)[0]].item())
            raise AssertionError(
                f"{label} batch {batch_idx} has valid slot beyond its seqlen={seqlen} at token {pos}."
            )
        values = row[valid_pos]
        if int(values.min().item()) < 0 or int(values.max().item()) >= args.cache_size:
            raise AssertionError(
                f"{label} batch {batch_idx} slot range is "
                f"[{int(values.min().item())}, {int(values.max().item())}], expected [0, {args.cache_size})."
            )
        sorted_values = torch.sort(values).values
        if not torch.equal(sorted_values, expected_slots):
            mismatch = (sorted_values != expected_slots).nonzero(as_tuple=False)
            pos = int(mismatch[0].item()) if mismatch.numel() else -1
            raise AssertionError(
                f"{label} batch {batch_idx} slot values are not exactly 0..cache_size-1; "
                f"first mismatch at sorted position {pos}."
            )


def prepare_reference_and_cache(args, tensors):
    reference = run_decode(tensors)
    torch.npu.synchronize()
    reference_cpu = reference.detach().cpu()
    cache_slots_cpu, target_miss_count = build_cache_slots(args, reference_cpu, tensors["seqlens"])
    check_cache_exact(args, cache_slots_cpu, tensors["seqlens"], "old_cache_slots")
    tensors["reference_sparse_indices"] = reference
    tensors["reference_sparse_indices_cpu"] = reference_cpu
    tensors["cache_slots_original_cpu"] = cache_slots_cpu
    tensors["cache_slots_original"] = cache_slots_cpu.to(tensors["cache_slots"].device)
    tensors["cache_slots"].copy_(tensors["cache_slots_original"])
    tensors["target_miss_count"] = target_miss_count
    torch.npu.synchronize()


def tensors_with_cache(tensors, cache_slots):
    local_tensors = dict(tensors)
    local_tensors["cache_slots"] = cache_slots
    return local_tensors


def check_decode_reference(args, output_cpu, seqlens):
    expected_shape = (args.bs, 1, SPARSE_COUNT)
    if tuple(output_cpu.shape) != expected_shape:
        raise AssertionError(f"decode sparse_indices shape {tuple(output_cpu.shape)} != {expected_shape}.")
    if output_cpu.dtype != torch.int32:
        raise AssertionError(f"decode sparse_indices dtype {output_cpu.dtype} != torch.int32.")
    for batch_idx, seqlen in enumerate(seqlens):
        row = output_cpu[batch_idx, 0]
        min_index = int(row.min().item())
        max_index = int(row.max().item())
        if min_index < 0 or max_index >= seqlen:
            raise AssertionError(
                f"decode batch {batch_idx} sparse_indices range [{min_index}, {max_index}] "
                f"is outside [0, {seqlen})."
            )
    print("decode_reference_check=passed")


def check_output_shapes(args, output):
    topk_index, topk_slots, miss_count = output
    expected_topk = (args.bs, 1, SPARSE_COUNT)
    if tuple(topk_index.shape) != expected_topk:
        raise AssertionError(f"topk_index shape {tuple(topk_index.shape)} != {expected_topk}.")
    if tuple(topk_slots.shape) != expected_topk:
        raise AssertionError(f"topk_slots shape {tuple(topk_slots.shape)} != {expected_topk}.")
    if tuple(miss_count.shape) != (args.bs,):
        raise AssertionError(f"miss_count shape {tuple(miss_count.shape)} != {(args.bs,)}.")
    for name, tensor in zip(("topk_index", "topk_slots", "miss_count"), output):
        if tensor.dtype != torch.int32:
            raise AssertionError(f"{name} dtype {tensor.dtype} != torch.int32.")


def check_update_behavior(args, tensors, output_cpu, new_cache_cpu, label):
    topk_index_cpu, topk_slots_cpu, miss_count_cpu = output_cpu
    reference_cpu = tensors["reference_sparse_indices_cpu"]
    old_cache_cpu = tensors["cache_slots_original_cpu"]
    target_miss_count = tensors["target_miss_count"]
    check_cache_exact(args, new_cache_cpu, tensors["seqlens"], f"{label} new_cache_slots")

    actual_counts = []
    for batch_idx in range(args.bs):
        indices = topk_index_cpu[batch_idx, 0].to(torch.long)
        slots = topk_slots_cpu[batch_idx, 0]
        miss_count = int(miss_count_cpu[batch_idx].item())
        expected_miss_count = int(target_miss_count[batch_idx].item())
        actual_counts.append(miss_count)

        actual_sorted = torch.sort(indices).values
        expected_sorted = torch.sort(reference_cpu[batch_idx, 0].to(torch.long)).values
        if not torch.equal(actual_sorted, expected_sorted):
            mismatch = (actual_sorted != expected_sorted).nonzero(as_tuple=False)
            pos = int(mismatch[0].item()) if mismatch.numel() else -1
            raise AssertionError(f"{label} batch {batch_idx} topk_index multiset mismatch at sorted position {pos}.")

        if bool((slots < 0).any().item()) or bool((slots >= args.cache_size).any().item()):
            bad = ((slots < 0) | (slots >= args.cache_size)).nonzero(as_tuple=False)
            pos = int(bad[0].item())
            raise AssertionError(
                f"{label} batch {batch_idx} topk_slots[{pos}]={int(slots[pos].item())}, "
                f"expected range [0, {args.cache_size})."
            )

        new_slots_by_index = new_cache_cpu[batch_idx].gather(0, indices)
        if not torch.equal(new_slots_by_index, slots):
            mismatch = (new_slots_by_index != slots).nonzero(as_tuple=False)
            pos = int(mismatch[0].item())
            token = int(indices[pos].item())
            raise AssertionError(
                f"{label} batch {batch_idx} new_cache_slots[{token}]={int(new_slots_by_index[pos].item())}, "
                f"topk_slots[{pos}]={int(slots[pos].item())}."
            )

        old_slots_by_index = old_cache_cpu[batch_idx].gather(0, indices)
        old_miss_count = int((old_slots_by_index == -1).sum().item())
        if miss_count != expected_miss_count or miss_count != old_miss_count:
            raise AssertionError(
                f"{label} batch {batch_idx} miss_count={miss_count}, "
                f"target={expected_miss_count}, old_topk_miss_count={old_miss_count}."
            )

        if miss_count > 0 and bool((old_slots_by_index[:miss_count] != -1).any().item()):
            bad = (old_slots_by_index[:miss_count] != -1).nonzero(as_tuple=False)
            pos = int(bad[0].item())
            token = int(indices[pos].item())
            raise AssertionError(
                f"{label} batch {batch_idx} old_cache_slots[{token}] should be -1 in miss prefix, "
                f"got {int(old_slots_by_index[pos].item())}."
            )

        if miss_count < SPARSE_COUNT:
            hit_old_slots = old_slots_by_index[miss_count:]
            hit_output_slots = slots[miss_count:]
            if bool((hit_old_slots == -1).any().item()):
                bad = (hit_old_slots == -1).nonzero(as_tuple=False)
                pos = int(bad[0].item()) + miss_count
                token = int(indices[pos].item())
                raise AssertionError(f"{label} batch {batch_idx} old_cache_slots[{token}] is -1 in hit suffix.")
            if not torch.equal(hit_old_slots, hit_output_slots):
                mismatch = (hit_old_slots != hit_output_slots).nonzero(as_tuple=False)
                pos = int(mismatch[0].item()) + miss_count
                token = int(indices[pos].item())
                raise AssertionError(
                    f"{label} batch {batch_idx} hit suffix mismatch at token {token}: "
                    f"old={int(old_slots_by_index[pos].item())}, out={int(slots[pos].item())}."
                )

    print(f"{label}_topk_index_multiset_check=passed")
    print(f"{label}_topk_slots_range_check=passed")
    print(f"{label}_topk_slots_new_cache_check=passed")
    print(f"{label}_old_cache_miss_prefix_check=passed")
    print(f"{label}_old_cache_hit_suffix_check=passed")
    print(f"{label}_new_cache_unique_range_check=passed")
    print(f"{label}_miss_count_check=passed")
    print(f"{label}_miss_count={actual_counts}")


def validate_update_op(args, tensors, runner, label):
    tensors["cache_slots"].copy_(tensors["cache_slots_original"])
    output = runner(tensors)
    torch.npu.synchronize()
    check_output_shapes(args, output)
    output_cpu = tuple(t.detach().cpu() for t in output)
    new_cache_cpu = tensors["cache_slots"].detach().cpu()
    check_update_behavior(args, tensors, output_cpu, new_cache_cpu, label)
    print(f"{label}_behavior_check=passed")
    return output


def benchmark_runner(args, tensors, runner):
    output = None
    times_ms = []
    for _ in range(args.warmup):
        tensors["cache_slots"].copy_(tensors["cache_slots_original"])
        output = runner(tensors)
    torch.npu.synchronize()

    try:
        for _ in range(args.iters):
            tensors["cache_slots"].copy_(tensors["cache_slots_original"])
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            output = runner(tensors)
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))
        return output, times_ms, "npu_event"
    except Exception as exc:
        print(f"[warn] NPU event timing failed for {runner.__name__}, falling back to wall clock: {exc}")

    output = None
    times_ms = []
    for _ in range(args.iters):
        tensors["cache_slots"].copy_(tensors["cache_slots_original"])
        torch.npu.synchronize()
        begin = time.perf_counter()
        output = runner(tensors)
        torch.npu.synchronize()
        times_ms.append((time.perf_counter() - begin) * 1000.0)
    return output, times_ms, "wall_clock"


def mean_us_int(times_ms):
    return int(statistics.mean(times_ms) * 1000.0)


def print_case_summary(args, tensors):
    target_head = tensors["target_miss_count"][: min(args.bs, 16)].tolist()
    seqlens = tensors["seqlens"]
    print(
        "case "
        f"bs={args.bs} min_seqlen={args.min_seqlen} max_seqlen={args.max_seqlen} "
        f"sampled_min_seqlen={min(seqlens)} sampled_max_seqlen={max(seqlens)} dtype={args.dtype} "
        f"cache_size={args.cache_size} min_miss_count={args.min_miss_count} "
        f"max_miss_count={args.max_miss_count}"
    )
    print(f"actual_seq_lengths_key={seqlens}")
    print(f"target_miss_count_head={target_head}")


def print_perf_summary(args, tensors, target_name, decode_times, target_times, timer_kind):
    target_head = tensors["target_miss_count"][: min(args.bs, 16)].tolist()
    seqlens = tensors["seqlens"]
    print(
        "summary "
        f"timer={timer_kind} bs={args.bs} min_seqlen={args.min_seqlen} max_seqlen={args.max_seqlen} "
        f"sampled_min_seqlen={min(seqlens)} sampled_max_seqlen={max(seqlens)} dtype={args.dtype} "
        f"cache_size={args.cache_size} min_miss_count={args.min_miss_count} "
        f"max_miss_count={args.max_miss_count} warmup={args.warmup} iters={args.iters} "
        f"li_avg_us={mean_us_int(decode_times)} "
        f"{target_name}_avg_us={mean_us_int(target_times)}"
    )
    print(f"target_miss_count_head={target_head}")


def prepare_case(args):
    validate_args(args)
    import_torch_npu()

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    tensors = allocate_inputs(args, device)
    prepare_reference_and_cache(args, tensors)
    print_case_summary(args, tensors)
    return tensors


def run_correctness(args):
    tensors = prepare_case(args)
    check_decode_reference(args, tensors["reference_sparse_indices_cpu"], tensors["seqlens"])
    check_tensors = tensors_with_cache(tensors, tensors["cache_slots_original"].clone())
    validate_update_op(args, check_tensors, run_decode_update, "update")
    print("all_update_correctness_checks=passed")


def run_perf(args):
    tensors = prepare_case(args)

    decode_tensors = tensors_with_cache(tensors, tensors["cache_slots_original"].clone())
    target_tensors = tensors_with_cache(tensors, tensors["cache_slots_original"].clone())
    _, decode_times, decode_timer = benchmark_runner(args, decode_tensors, run_decode)
    _, target_times, target_timer = benchmark_runner(args, target_tensors, run_decode_update)
    timer_kind = decode_timer if decode_timer == target_timer else "mixed"
    print_perf_summary(args, tensors, "li_update", decode_times, target_times, timer_kind)
