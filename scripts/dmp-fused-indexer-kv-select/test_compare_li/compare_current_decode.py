#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import argparse
import sys
from pathlib import Path


SPARSE_COUNT = 2048
DEFAULT_INPUT = Path(__file__).resolve().parent / "lightning_indexer_reference.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the current LightningIndexerDecode and compare it with a saved vLLM reference."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def import_current_op():
    import torch
    import torch_npu  # noqa: F401

    repo_root = Path(__file__).resolve().parents[1]
    extension_dir = repo_root / "torch_extension"
    sys.path.insert(0, str(extension_dir))
    import lightning_indexer_decode_custom_ops  # noqa: F401

    namespace = getattr(torch.ops, "custom", None)
    op = getattr(namespace, "npu_lightning_indexer_decode", None) if namespace is not None else None
    if op is None:
        raise RuntimeError("torch.ops.custom.npu_lightning_indexer_decode is not registered.")
    return torch, op


def validate_saved_data(torch, payload):
    required = {
        "metadata",
        "query",
        "key",
        "weights",
        "actual_seq_lengths_key",
        "block_table",
        "reference_topk_index",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Saved reference is missing keys: {missing}.")

    metadata = payload["metadata"]
    bs = int(metadata["bs"])
    seqlens = [int(value) for value in metadata["seqlens"]]
    if len(seqlens) != bs or len(set(seqlens)) != bs:
        raise ValueError("Saved sequence lengths must be distinct and match bs.")
    if int(metadata["sparse_count"]) != SPARSE_COUNT:
        raise ValueError("Saved sparse_count must be 2048.")
    if tuple(payload["reference_topk_index"].shape) != (bs, 1, SPARSE_COUNT):
        raise ValueError("Saved reference_topk_index has an invalid shape.")
    if payload["reference_topk_index"].dtype != torch.int32:
        raise ValueError("Saved reference_topk_index must be int32.")
    return metadata, seqlens


def validate_current(torch, output, seqlens):
    expected_shape = (len(seqlens), 1, SPARSE_COUNT)
    if tuple(output.shape) != expected_shape:
        raise AssertionError(f"current shape {tuple(output.shape)} != {expected_shape}.")
    if output.dtype != torch.int32:
        raise AssertionError(f"current dtype {output.dtype} != torch.int32.")
    output_cpu = output.detach().cpu()
    for batch_idx, seqlen in enumerate(seqlens):
        row = output_cpu[batch_idx, 0]
        min_index = int(row.min().item())
        max_index = int(row.max().item())
        if min_index < 0 or max_index >= seqlen:
            raise AssertionError(
                f"current batch {batch_idx} index range [{min_index}, {max_index}] "
                f"is outside [0, {seqlen})."
            )
        if torch.unique(row).numel() != SPARSE_COUNT:
            raise AssertionError(f"current batch {batch_idx} contains duplicate indices.")
    return output_cpu


def compare_outputs(torch, current_cpu, reference_cpu, seqlens):
    ordered_equal = 0
    for batch_idx, seqlen in enumerate(seqlens):
        current_row = current_cpu[batch_idx, 0]
        reference_row = reference_cpu[batch_idx, 0]
        if torch.equal(current_row, reference_row):
            ordered_equal += 1

        current_sorted = torch.sort(current_row).values
        reference_sorted = torch.sort(reference_row).values
        if not torch.equal(current_sorted, reference_sorted):
            current_set = set(current_row.tolist())
            reference_set = set(reference_row.tolist())
            only_current = sorted(current_set - reference_set)[:16]
            only_reference = sorted(reference_set - current_set)[:16]
            raise AssertionError(
                f"batch {batch_idx}, seqlen={seqlen}: top2048 multiset mismatch; "
                f"only_current={only_current}, only_reference={only_reference}."
            )
    print("topk_index_multiset_check=passed")
    print(f"ordered_equal_batches={ordered_equal}/{len(seqlens)}")


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Reference data file does not exist: {args.input}")

    torch, current_op = import_current_op()
    payload = torch.load(args.input, map_location="cpu")
    metadata, seqlens = validate_saved_data(torch, payload)

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    query = payload["query"].to(device)
    key = payload["key"].to(device)
    weights = payload["weights"].to(device)
    actual_seq_lengths_key = payload["actual_seq_lengths_key"].to(device)
    block_table = payload["block_table"].to(device)

    print(
        f"case bs={metadata['bs']} dtype={metadata['dtype']} block_size={metadata['block_size']} "
        f"min_seqlen={min(seqlens)} max_seqlen={max(seqlens)} "
        f"num_blocks={metadata['num_blocks']} seed={metadata['seed']}"
    )
    print(f"actual_seq_lengths_key={seqlens}")

    current_output = current_op(query, key, weights, actual_seq_lengths_key, block_table)
    torch.npu.synchronize()
    current_cpu = validate_current(torch, current_output, seqlens)
    reference_cpu = payload["reference_topk_index"]
    print("current_output_check=passed")
    compare_outputs(torch, current_cpu, reference_cpu, seqlens)
    print("lightning_indexer_decode_alignment_check=passed")


if __name__ == "__main__":
    main()
