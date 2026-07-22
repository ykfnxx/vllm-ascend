#!/usr/bin/env python3

import argparse
import statistics


Q_HEADS = 64
K_HEADS = 1
HEAD_DIM = 128
SPARSE_COUNT = 2048


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the vLLM-Ascend LightningIndexer operator.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[24, 48])
    parser.add_argument("--seqlens", type=int, nargs="+", default=[65536, 131072])
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def validate_args(args):
    if args.device < 0:
        raise ValueError("device must be non-negative.")
    if any(bs <= 0 for bs in args.batch_sizes):
        raise ValueError("batch sizes must be positive.")
    if any(seqlen < SPARSE_COUNT for seqlen in args.seqlens):
        raise ValueError("every seqlen must be at least 2048.")
    if args.block_size != 128:
        raise ValueError("LightningIndexer PA_BSND performance comparison requires block_size=128.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive.")


def import_indexer():
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


def allocate_case(torch, args, device, bs, seqlen):
    case_seed = args.seed + bs * 1000003 + seqlen
    torch.manual_seed(case_seed)
    torch.npu.manual_seed_all(case_seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    blocks_per_batch = (seqlen + args.block_size - 1) // args.block_size
    num_blocks = 1 + bs * blocks_per_batch

    query = torch.empty((bs, Q_HEADS, HEAD_DIM), dtype=dtype, device=device)
    key = torch.empty((num_blocks, args.block_size, K_HEADS, HEAD_DIM), dtype=dtype, device=device)
    weights = torch.empty((bs, Q_HEADS), dtype=dtype, device=device)
    query.uniform_(-1.0, 1.0)
    key.uniform_(-1.0, 1.0)
    weights.uniform_(-1.0, 1.0)

    return {
        "query": query,
        "key": key,
        "weights": weights,
        "actual_seq_lengths_query": torch.arange(1, bs + 1, dtype=torch.int32, device=device),
        "actual_seq_lengths_key": torch.full((bs,), seqlen, dtype=torch.int32, device=device),
        "block_table": torch.arange(1, num_blocks, dtype=torch.int32, device=device).reshape(
            bs, blocks_per_batch
        ),
    }


def run_indexer(op, tensors):
    output = op(
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
    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise RuntimeError(f"Expected one LightningIndexer output, got {len(output)}.")
        output = output[0]
    return output


def benchmark_case(torch, op, tensors, bs, seqlen, warmup, iters):
    output = run_indexer(op, tensors)
    torch.npu.synchronize()
    if tuple(output.shape) != (bs, 1, SPARSE_COUNT) or output.dtype != torch.int32:
        raise RuntimeError(f"Unexpected output: shape={tuple(output.shape)}, dtype={output.dtype}.")
    if int(output.min().item()) < 0 or int(output.max().item()) >= seqlen:
        raise RuntimeError("LightningIndexer returned an out-of-range token index.")

    for _ in range(warmup):
        output = run_indexer(op, tensors)
    torch.npu.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        output = run_indexer(op, tensors)
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return int(round(statistics.mean(times_ms) * 1000.0))


def main():
    args = parse_args()
    validate_args(args)
    torch, op = import_indexer()
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    results = []
    for bs in args.batch_sizes:
        for seqlen in args.seqlens:
            tensors = allocate_case(torch, args, device, bs, seqlen)
            avg_us = benchmark_case(torch, op, tensors, bs, seqlen, args.warmup, args.iters)
            results.append((bs, seqlen, avg_us))
            print(f"summary op=LightningIndexer bs={bs} seqlen={seqlen} avg_us={avg_us}")
            del tensors
            torch.npu.empty_cache()

    print("\nLightningIndexer")
    print(f"{'bs':>6} {'seqlen':>10} {'avg_us':>10}")
    for bs, seqlen, avg_us in results:
        print(f"{bs:>6} {seqlen:>10} {avg_us:>10}")


if __name__ == "__main__":
    main()
