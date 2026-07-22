#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import argparse
import random
from pathlib import Path


SPARSE_COUNT = 2048
Q_HEADS = 64
K_HEADS = 1
HEAD_DIM = 128
MAX_SEQLEN = 262144
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "lightning_indexer_reference.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate inputs and the vLLM Ascend LightningIndexer reference output."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bs", type=int, default=24)
    parser.add_argument("--min-seqlen", type=int, default=32768)
    parser.add_argument("--max-seqlen", type=int, default=65536)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_args(args):
    if args.bs <= 0:
        raise ValueError("bs must be positive.")
    if args.min_seqlen < SPARSE_COUNT:
        raise ValueError("min-seqlen must be >= 2048.")
    if args.max_seqlen > MAX_SEQLEN:
        raise ValueError("max-seqlen must be <= 262144.")
    if args.min_seqlen > args.max_seqlen:
        raise ValueError("min-seqlen must be <= max-seqlen.")
    if args.bs > 1 and args.max_seqlen - args.min_seqlen < args.bs - 1:
        raise ValueError("The seqlen range must contain at least bs distinct values.")
    if args.block_size <= 0 or args.block_size > 1024 or args.block_size % 16 != 0:
        raise ValueError("block-size must be a multiple of 16 in (0, 1024].")


def make_distinct_seqlens(args):
    if args.bs == 1:
        lengths = [args.max_seqlen]
    else:
        span = args.max_seqlen - args.min_seqlen
        lengths = [args.min_seqlen + index * span // (args.bs - 1) for index in range(args.bs)]
    random.Random(args.seed).shuffle(lengths)
    if len(set(lengths)) != args.bs:
        raise RuntimeError("Failed to construct distinct sequence lengths.")
    return lengths


def import_reference_op():
    import torch
    import torch_npu  # noqa: F401
    import vllm  # noqa: F401
    import vllm_ascend  # noqa: F401
    from vllm_ascend.ops.layer_shard_linear import (  # noqa: F401
        is_hidden_layer,
        post_process_after_loading_for_shard_weight_series,
        reach_layer_for_shard_weight_series,
        register_all_layers_to_shard_weight_series,
    )
    from vllm_ascend.platform import NPUPlatform

    NPUPlatform.import_kernels()
    from vllm_ascend import vllm_ascend_C  # noqa: F401

    namespace = getattr(torch.ops, "_C_ascend", None)
    op = getattr(namespace, "npu_lightning_indexer", None) if namespace is not None else None
    if op is None:
        raise RuntimeError("torch.ops._C_ascend.npu_lightning_indexer is not registered.")
    return torch, op


def allocate_inputs(torch, args, device):
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    seqlens = make_distinct_seqlens(args)
    block_counts = [(length + args.block_size - 1) // args.block_size for length in seqlens]
    max_blocks_per_batch = max(block_counts)
    total_active_blocks = sum(block_counts)

    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(args.seed + 1)
    physical_ids = torch.randperm(total_active_blocks, generator=cpu_generator, dtype=torch.int64) + 1
    block_table_cpu = torch.zeros((args.bs, max_blocks_per_batch), dtype=torch.int32)
    cursor = 0
    for batch_idx, block_count in enumerate(block_counts):
        block_table_cpu[batch_idx, :block_count] = physical_ids[cursor:cursor + block_count].to(torch.int32)
        cursor += block_count

    num_blocks = total_active_blocks + 1
    query = torch.empty((args.bs, Q_HEADS, HEAD_DIM), dtype=dtype, device=device)
    key = torch.empty((num_blocks, args.block_size, K_HEADS, HEAD_DIM), dtype=dtype, device=device)
    weights = torch.empty((args.bs, Q_HEADS), dtype=dtype, device=device)
    query.uniform_(-1.0, 1.0)
    key.uniform_(-1.0, 1.0)
    weights.uniform_(-1.0, 1.0)

    return {
        "query": query,
        "key": key,
        "weights": weights,
        "actual_seq_lengths_query": torch.arange(1, args.bs + 1, dtype=torch.int32, device=device),
        "actual_seq_lengths_key": torch.tensor(seqlens, dtype=torch.int32, device=device),
        "block_table": block_table_cpu.to(device),
        "seqlens": seqlens,
        "block_counts": block_counts,
        "num_blocks": num_blocks,
    }


def validate_reference(torch, output, seqlens):
    expected_shape = (len(seqlens), 1, SPARSE_COUNT)
    if tuple(output.shape) != expected_shape:
        raise AssertionError(f"reference shape {tuple(output.shape)} != {expected_shape}.")
    if output.dtype != torch.int32:
        raise AssertionError(f"reference dtype {output.dtype} != torch.int32.")
    output_cpu = output.detach().cpu()
    for batch_idx, seqlen in enumerate(seqlens):
        row = output_cpu[batch_idx, 0]
        min_index = int(row.min().item())
        max_index = int(row.max().item())
        if min_index < 0 or max_index >= seqlen:
            raise AssertionError(
                f"reference batch {batch_idx} index range [{min_index}, {max_index}] "
                f"is outside [0, {seqlen})."
            )
        if torch.unique(row).numel() != SPARSE_COUNT:
            raise AssertionError(f"reference batch {batch_idx} contains duplicate indices.")
    return output_cpu


def main():
    args = parse_args()
    validate_args(args)
    torch, reference_op = import_reference_op()
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    tensors = allocate_inputs(torch, args, device)

    print(
        f"case bs={args.bs} dtype={args.dtype} block_size={args.block_size} "
        f"min_seqlen={min(tensors['seqlens'])} max_seqlen={max(tensors['seqlens'])} "
        f"num_blocks={tensors['num_blocks']} seed={args.seed}"
    )
    print(f"actual_seq_lengths_key={tensors['seqlens']}")

    reference_output = reference_op(
        query=tensors["query"],
        key=tensors["key"],
        weights=tensors["weights"],
        actual_seq_lengths_query=tensors["actual_seq_lengths_query"],
        actual_seq_lengths_key=tensors["actual_seq_lengths_key"],
        block_table=tensors["block_table"],
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
    )
    torch.npu.synchronize()
    reference_cpu = validate_reference(torch, reference_output, tensors["seqlens"])

    payload = {
        "metadata": {
            "bs": args.bs,
            "dtype": args.dtype,
            "block_size": args.block_size,
            "sparse_count": SPARSE_COUNT,
            "seed": args.seed,
            "seqlens": tensors["seqlens"],
            "block_counts": tensors["block_counts"],
            "num_blocks": tensors["num_blocks"],
        },
        "query": tensors["query"].detach().cpu(),
        "key": tensors["key"].detach().cpu(),
        "weights": tensors["weights"].detach().cpu(),
        "actual_seq_lengths_query": tensors["actual_seq_lengths_query"].detach().cpu(),
        "actual_seq_lengths_key": tensors["actual_seq_lengths_key"].detach().cpu(),
        "block_table": tensors["block_table"].detach().cpu(),
        "reference_topk_index": reference_cpu,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    size_gib = args.output.stat().st_size / (1024 ** 3)
    print("reference_output_check=passed")
    print(f"saved={args.output} size_gib={size_gib:.3f}")


if __name__ == "__main__":
    main()
