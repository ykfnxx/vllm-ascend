#!/usr/bin/env python3
# coding=utf-8
"""DBA-style pipeline: LightningIndexer + MockKVSelect + KVGather + segmented SFA.

This script keeps the current branch's real segmented-SFA data path:

  real KVSelect (setup only) -> KVGather plan + hit/miss sparse indices

The timed path replaces KVSelect with the AICPU MockKVSelect op.  KVGather and
segmented SFA consume the setup-time real KVSelect plan, so the measurement
isolates the scheduling benefit of moving KVSelect-like latency off AI Core.

Overall batch size is split into two half batches:

  1. indexer0
  2. indexer1 || mock_kv_select0
  3. mock_kv_select1 || hit_sfa0, while gather0 can start after select0
  4. gather1 after select1
  5. miss_sfa after the corresponding gather and hit_sfa complete
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch_npu

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent
REPO_SRC = THIS_DIR.parent.parent / "src"
TORCH_OPS_EXTENSION_DIR = THIS_DIR.parent.parent / "op" / "torch_ops_extension"
for _path in (THIS_DIR, EXPERIMENTS_DIR / "lightning_indexer", REPO_SRC, TORCH_OPS_EXTENSION_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("USE_CUSTOM_SFA", "1")

from test_npu_gather_selection_kv_cache_perf import (  # noqa: E402
    HEAD_NUM,
    KV_CACHE_DIM,
    SELECTION_TOPK,
    SELECTION_TOPK_BLOCK_SIZE,
    SEQ_LEN,
    make_inputs,
    random_selection_topk,
    run_gather,
)
from test_npu_kv_select_gather_correctness import make_next_topk, set_topk  # noqa: E402
from test_npu_kv_select_gather_perf import make_workspace, run_kv_gather, run_kv_select  # noqa: E402
from test_npu_segmented_sfa import (  # noqa: E402
    HitMissCounts,
    SelectionSfaContext,
    compact_valid_indices,
    compute_selection_actual_seq,
    make_query,
)
from dual_attention import run_sparse_flash_attention  # noqa: E402

try:
    import custom_ops  # noqa: F401,E402
except ImportError as exc:
    raise SystemExit("custom_ops required; build op/torch_ops_extension first.") from exc


DEFAULT_WARMUP = 2
DEFAULT_ITERS = 5
DEFAULT_FULL_BATCH = 64
DEFAULT_MAX_SEQ_LEN = 65_536
DEFAULT_REUSE_RATE = 0.9
DEFAULT_MOCK_WAIT_US = 25
DEFAULT_GATHER_CUBE_CORES = 8
DEFAULT_GATHER_VECTOR_CORES = 16
DEFAULT_HIT_SFA_CUBE_CORES = 16
DEFAULT_HIT_SFA_VECTOR_CORES = 32
DEFAULT_TRACE_DIR = THIS_DIR / "profiler_trace/mock_kvselect_segmented_pipeline"
DEFAULT_CSV = THIS_DIR / "mock_kvselect_segmented_pipeline_bs64_seq65536.csv"
MUTATED_INPUTS = {
    "selection_k_rope",
    "selection_kv_cache",
    "selection_kv_block_table",
    "selection_kv_block_status",
}
INDEXER_BLOCK_SIZE = 128
INDEXER_SPARSE_COUNT = 2048
INDEXER_SPARSE_MODE = 3
INDEXER_HEADS = 64
INDEXER_HEAD_DIM = 128
INDEXER_DTYPE = torch.bfloat16


class IndexerInputs(NamedTuple):
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    q_lens: torch.Tensor
    k_lens: torch.Tensor
    block_table: torch.Tensor


class HalfRunState(NamedTuple):
    inputs: dict[str, torch.Tensor]
    workspace: dict[str, torch.Tensor]
    mock_workspace: dict[str, torch.Tensor]
    context: SelectionSfaContext
    actual_seq_query: torch.Tensor
    hit_softmax_max: torch.Tensor
    hit_softmax_sum: torch.Tensor
    miss_softmax_max: torch.Tensor
    miss_softmax_sum: torch.Tensor
    hit_indices: torch.Tensor | None
    miss_indices: torch.Tensor | None
    counts: HitMissCounts


class HitState(NamedTuple):
    out: torch.Tensor
    softmax_max: torch.Tensor
    softmax_sum: torch.Tensor


class CapturedPipelineGraph(NamedTuple):
    graph: torch.npu.NPUGraph
    stream: torch.npu.Stream
    outputs: dict[str, torch.Tensor]


@dataclass(frozen=True)
class PreparedHalf:
    seed: int
    inputs: dict[str, torch.Tensor]
    workspace: dict[str, torch.Tensor]
    context: SelectionSfaContext
    actual_seq_query: torch.Tensor
    hit_indices: torch.Tensor | None
    miss_indices: torch.Tensor | None
    counts: HitMissCounts


@dataclass(frozen=True)
class Streams:
    index0: torch.npu.Stream
    index1: torch.npu.Stream
    select0: torch.npu.Stream
    select1: torch.npu.Stream
    gather0: torch.npu.Stream
    gather1: torch.npu.Stream
    hit0: torch.npu.Stream
    hit1: torch.npu.Stream
    tail: torch.npu.Stream

    def synchronize(self) -> None:
        for stream in self:
            stream.synchronize()

    def __iter__(self):
        return iter(
            (
                self.index0,
                self.index1,
                self.select0,
                self.select1,
                self.gather0,
                self.gather1,
                self.hit0,
                self.hit1,
                self.tail,
            )
        )


@dataclass(frozen=True)
class ThreeStreams:
    compute: torch.npu.Stream
    select: torch.npu.Stream
    gather: torch.npu.Stream

    def synchronize(self) -> None:
        for stream in self:
            stream.synchronize()

    def __iter__(self):
        return iter((self.compute, self.select, self.gather))


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.npu.current_device())


def make_stream(device: torch.device) -> torch.npu.Stream:
    try:
        return torch.npu.Stream(device=device)
    except TypeError:
        return torch.npu.Stream()


def set_stream_limit(stream: torch.npu.Stream, cube_cores: int, vector_cores: int) -> dict:
    torch.npu.set_stream_limit(stream, cube_cores, vector_cores)
    try:
        return torch.npu.get_stream_limit(stream)
    except Exception:
        return {}


def make_streams(device: torch.device, args: argparse.Namespace) -> Streams:
    streams = Streams(
        index0=make_stream(device),
        index1=make_stream(device),
        select0=make_stream(device),
        select1=make_stream(device),
        gather0=make_stream(device),
        gather1=make_stream(device),
        hit0=make_stream(device),
        hit1=make_stream(device),
        tail=make_stream(device),
    )
    limits = {
        "gather0": set_stream_limit(streams.gather0, args.gather_cube_cores, args.gather_vector_cores),
        "gather1": set_stream_limit(streams.gather1, args.gather_cube_cores, args.gather_vector_cores),
        "hit0": set_stream_limit(streams.hit0, args.hit_sfa_cube_cores, args.hit_sfa_vector_cores),
        "hit1": set_stream_limit(streams.hit1, args.hit_sfa_cube_cores, args.hit_sfa_vector_cores),
    }
    for name, limit in limits.items():
        if limit:
            print(f"{name} stream limit: {limit}", flush=True)
    return streams


def make_three_streams(device: torch.device, args: argparse.Namespace) -> ThreeStreams:
    streams = ThreeStreams(
        compute=make_stream(device),
        select=make_stream(device),
        gather=make_stream(device),
    )
    limit = set_stream_limit(streams.gather, args.gather_cube_cores, args.gather_vector_cores)
    if limit:
        print(f"gather stream limit: {limit}", flush=True)
    return streams


def clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() if name in MUTATED_INPUTS else tensor for name, tensor in inputs.items()}


def clone_workspace(workspace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in workspace.items()}


def make_mock_workspace(workspace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: torch.empty_like(tensor)
        for name, tensor in workspace.items()
        if name != "selection_kv_actual_seq"
    }


def materialize_half(prepared: PreparedHalf) -> HalfRunState:
    inputs = clone_inputs(prepared.inputs)
    workspace = clone_workspace(prepared.workspace)
    context = SelectionSfaContext(
        query=prepared.context.query,
        query_rope=prepared.context.query_rope,
        inputs=inputs,
        actual_seq=prepared.context.actual_seq,
    )
    softmax_shape = (context.query.shape[0], context.query.shape[1])
    neg_inf = torch.finfo(torch.float32).min
    hit_softmax_max = torch.full(softmax_shape, neg_inf, dtype=torch.float32, device=context.query.device)
    hit_softmax_sum = torch.zeros(softmax_shape, dtype=torch.float32, device=context.query.device)
    miss_softmax_max = torch.full(softmax_shape, neg_inf, dtype=torch.float32, device=context.query.device)
    miss_softmax_sum = torch.zeros(softmax_shape, dtype=torch.float32, device=context.query.device)
    torch.npu.current_stream().synchronize()
    return HalfRunState(
        inputs=inputs,
        workspace=workspace,
        mock_workspace=make_mock_workspace(workspace),
        context=context,
        actual_seq_query=prepared.actual_seq_query,
        hit_softmax_max=hit_softmax_max,
        hit_softmax_sum=hit_softmax_sum,
        miss_softmax_max=miss_softmax_max,
        miss_softmax_sum=miss_softmax_sum,
        hit_indices=prepared.hit_indices,
        miss_indices=prepared.miss_indices,
        counts=prepared.counts,
    )


def prepare_half(
    *,
    batch_size: int,
    max_seq_len: int,
    reuse_rate: float,
    device: torch.device,
    seed: int,
    offload_full_cache: bool,
) -> PreparedHalf:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    n_blocks = (max_seq_len + SELECTION_TOPK_BLOCK_SIZE - 1) // SELECTION_TOPK_BLOCK_SIZE
    topk = random_selection_topk(
        np.arange(0, n_blocks, dtype=np.int32), batch_size, SEQ_LEN, HEAD_NUM, SELECTION_TOPK
    )
    inputs = make_inputs(batch_size, max_seq_len, topk, device, offload_full_cache=offload_full_cache)

    # Prime the HBM selection cache so the next step has controlled hit/miss reuse.
    run_gather(inputs)
    torch.npu.synchronize()

    next_topk = make_next_topk(topk, batch_size, max_seq_len, reuse_rate, rng)
    set_topk(inputs, next_topk)

    workspace = make_workspace(inputs, batch_size, device)
    run_kv_select(inputs, workspace)
    torch.npu.synchronize()

    hit_indices, hit_total = compact_valid_indices(workspace["hit_sparse_indices"])
    miss_indices, miss_total = compact_valid_indices(workspace["miss_insert_indices"])
    query, query_rope = make_query(
        batch_size,
        device,
        inputs["selection_kv_cache"].dtype,
        seed + 1009,
    )
    context = SelectionSfaContext(
        query=query,
        query_rope=query_rope,
        inputs=inputs,
        actual_seq=compute_selection_actual_seq(inputs),
    )
    actual_seq_query = torch.arange(
        1,
        query.shape[0] + 1,
        dtype=torch.int32,
        device=device,
    )
    return PreparedHalf(
        seed=seed,
        inputs=inputs,
        workspace=workspace,
        context=context,
        actual_seq_query=actual_seq_query,
        hit_indices=hit_indices,
        miss_indices=miss_indices,
        counts=HitMissCounts(hit=hit_total, miss=miss_total),
    )


def make_indexer_inputs(
    batch_size: int,
    key_seq_len: int,
    device: torch.device,
    query_seq_len: int = 1,
) -> IndexerInputs:
    blocks_per_seq = (key_seq_len + INDEXER_BLOCK_SIZE - 1) // INDEXER_BLOCK_SIZE
    num_key_blocks = batch_size * blocks_per_seq
    token_count = batch_size * query_seq_len
    query = torch.randn(token_count, INDEXER_HEADS, INDEXER_HEAD_DIM, dtype=INDEXER_DTYPE, device=device)
    key = torch.randn(
        num_key_blocks,
        INDEXER_BLOCK_SIZE,
        1,
        INDEXER_HEAD_DIM,
        dtype=INDEXER_DTYPE,
        device=device,
    )
    weights = torch.randn(token_count, INDEXER_HEADS, dtype=INDEXER_DTYPE, device=device)
    q_lens = torch.cumsum(
        torch.full((batch_size,), query_seq_len, dtype=torch.int32, device=device),
        dim=0,
    ).to(torch.int32)
    k_lens = torch.full((batch_size,), key_seq_len, dtype=torch.int32, device=device)
    block_table = torch.arange(num_key_blocks, dtype=torch.int32, device=device).view(
        batch_size,
        blocks_per_seq,
    )
    return IndexerInputs(query, key, weights, q_lens, k_lens, block_table)


def run_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    q_lens: torch.Tensor,
    k_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    out = torch_npu.npu_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=q_lens,
        actual_seq_lengths_key=k_lens,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=INDEXER_SPARSE_COUNT,
        sparse_mode=INDEXER_SPARSE_MODE,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


def run_indexer_once(indexer_inputs) -> torch.Tensor:
    return run_indexer(*indexer_inputs)


def run_mock_kv_select(state: HalfRunState, mock_wait_us: int) -> None:
    ws = state.mock_workspace
    torch.ops.custom.npu_mock_kv_select_out(
        state.inputs["selection_k_rope"],
        state.inputs["selection_kv_cache"],
        state.inputs["selection_kv_block_table"],
        state.inputs["selection_kv_block_status"],
        state.inputs["selection_topk_indices"],
        state.inputs["full_k_rope"],
        state.inputs["full_kv_cache"],
        state.inputs["full_kv_block_table"],
        state.inputs["full_kv_actual_seq"],
        state.inputs["full_q_actual_seq"],
        ws["hit_sparse_indices"],
        ws["miss_topk_indices"],
        ws["miss_insert_indices"],
        ws["hit_actual_seq"],
        ws["miss_actual_seq"],
        ws["miss_count"],
        ws["hit_count"],
        ws["selection_status_empty"],
        selection_topk_block_size=SELECTION_TOPK_BLOCK_SIZE,
        mock_wait_us=mock_wait_us,
    )


def run_selection_sfa_prepared(
    state: HalfRunState,
    sparse_indices: torch.Tensor,
    *,
    softmax_max_out: torch.Tensor | None = None,
    softmax_sum_out: torch.Tensor | None = None,
) -> torch.Tensor:
    return run_sparse_flash_attention(
        state.context.query,
        state.inputs["selection_kv_cache"].unsqueeze(2),
        state.inputs["selection_kv_cache"].unsqueeze(2),
        sparse_indices,
        float(1.0 / np.sqrt(KV_CACHE_DIM)),
        query_rope=state.context.query_rope,
        key_rope=state.inputs["selection_k_rope"].unsqueeze(2),
        block_table=state.inputs["selection_kv_block_table"],
        actual_seq_lengths_query=state.actual_seq_query,
        actual_seq_lengths_kv=state.context.actual_seq,
        sparse_block_size=SELECTION_TOPK_BLOCK_SIZE,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        softmax_max_out=softmax_max_out,
        softmax_sum_out=softmax_sum_out,
    )


def run_hit_sfa(state: HalfRunState) -> HitState:
    if state.hit_indices is None:
        return HitState(
            state.context.query.new_zeros(state.context.query.shape),
            state.hit_softmax_max,
            state.hit_softmax_sum,
        )
    out = run_selection_sfa_prepared(
        state,
        state.hit_indices,
        softmax_max_out=state.hit_softmax_max,
        softmax_sum_out=state.hit_softmax_sum,
    )
    return HitState(out, state.hit_softmax_max, state.hit_softmax_sum)


def run_kv_gather_plan(state: HalfRunState) -> torch.Tensor:
    run_kv_gather(state.inputs, state.workspace)
    return state.workspace["selection_kv_actual_seq"]


def run_da_attention_merge(state: HalfRunState, hit: HitState, miss_out: torch.Tensor) -> torch.Tensor:
    return torch.ops.custom.npu_da_attention_merge(
        hit.out,
        hit.softmax_max,
        hit.softmax_sum,
        miss_out,
        state.miss_softmax_max,
        state.miss_softmax_sum,
    )


def run_miss_sfa_and_merge(state: HalfRunState, hit: HitState) -> torch.Tensor:
    if state.miss_indices is None:
        return hit.out

    if state.hit_indices is None:
        return run_selection_sfa_prepared(state, state.miss_indices)

    miss_out = run_selection_sfa_prepared(
        state,
        state.miss_indices,
        softmax_max_out=state.miss_softmax_max,
        softmax_sum_out=state.miss_softmax_sum,
    )
    return run_da_attention_merge(state, hit, miss_out)


def run_serial_once(
    *,
    stream: torch.npu.Stream,
    indexer_inputs,
    half0: HalfRunState,
    half1: HalfRunState,
    mock_wait_us: int,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    with torch.npu.stream(stream):
        outputs["indexer0"] = run_indexer_once(indexer_inputs)
        run_mock_kv_select(half0, mock_wait_us)
        hit0 = run_hit_sfa(half0)
        outputs["gather0_actual_seq"] = run_kv_gather_plan(half0)
        outputs["attn0"] = run_miss_sfa_and_merge(half0, hit0)

        outputs["indexer1"] = run_indexer_once(indexer_inputs)
        run_mock_kv_select(half1, mock_wait_us)
        hit1 = run_hit_sfa(half1)
        outputs["gather1_actual_seq"] = run_kv_gather_plan(half1)
        outputs["attn1"] = run_miss_sfa_and_merge(half1, hit1)
    stream.synchronize()
    return outputs


def run_pipeline_once(
    *,
    streams: Streams,
    indexer_inputs,
    half0: HalfRunState,
    half1: HalfRunState,
    mock_wait_us: int,
    synchronize: bool = True,
) -> dict[str, torch.Tensor]:
    start = torch.npu.Event(enable_timing=False)
    indexer0_done = torch.npu.Event(enable_timing=False)
    indexer1_done = torch.npu.Event(enable_timing=False)
    select0_done = torch.npu.Event(enable_timing=False)
    select1_done = torch.npu.Event(enable_timing=False)
    gather0_done = torch.npu.Event(enable_timing=False)
    gather1_done = torch.npu.Event(enable_timing=False)
    hit0_done = torch.npu.Event(enable_timing=False)
    hit1_done = torch.npu.Event(enable_timing=False)
    tail_done = torch.npu.Event(enable_timing=False)
    outputs: dict[str, torch.Tensor] = {}

    start.record(torch.npu.current_stream())
    with torch.npu.stream(streams.index0):
        streams.index0.wait_event(start)
        outputs["indexer0"] = run_indexer_once(indexer_inputs)
        indexer0_done.record()

    with torch.npu.stream(streams.index1):
        streams.index1.wait_event(indexer0_done)
        outputs["indexer1"] = run_indexer_once(indexer_inputs)
        indexer1_done.record()

    with torch.npu.stream(streams.select0):
        streams.select0.wait_event(indexer0_done)
        run_mock_kv_select(half0, mock_wait_us)
        select0_done.record()

    with torch.npu.stream(streams.gather0):
        streams.gather0.wait_event(select0_done)
        outputs["gather0_actual_seq"] = run_kv_gather_plan(half0)
        gather0_done.record()

    with torch.npu.stream(streams.hit0):
        streams.hit0.wait_event(select0_done)
        hit0 = run_hit_sfa(half0)
        hit0_done.record()

    with torch.npu.stream(streams.select1):
        streams.select1.wait_event(indexer1_done)
        run_mock_kv_select(half1, mock_wait_us)
        select1_done.record()

    with torch.npu.stream(streams.gather1):
        streams.gather1.wait_event(select1_done)
        outputs["gather1_actual_seq"] = run_kv_gather_plan(half1)
        gather1_done.record()

    with torch.npu.stream(streams.hit1):
        streams.hit1.wait_event(select1_done)
        hit1 = run_hit_sfa(half1)
        hit1_done.record()

    with torch.npu.stream(streams.tail):
        streams.tail.wait_event(gather0_done)
        streams.tail.wait_event(hit0_done)
        outputs["attn0"] = run_miss_sfa_and_merge(half0, hit0)
        streams.tail.wait_event(gather1_done)
        streams.tail.wait_event(hit1_done)
        outputs["attn1"] = run_miss_sfa_and_merge(half1, hit1)
        tail_done.record()

    torch.npu.current_stream().wait_event(tail_done)
    if synchronize:
        for stream in streams:
            stream.synchronize()
    return outputs


def run_pipeline_three_stream_once(
    *,
    streams: ThreeStreams,
    indexer_inputs,
    half0: HalfRunState,
    half1: HalfRunState,
    mock_wait_us: int,
    synchronize: bool = True,
) -> dict[str, torch.Tensor]:
    start = torch.npu.Event(enable_timing=False)
    indexer0_done = torch.npu.Event(enable_timing=False)
    indexer1_done = torch.npu.Event(enable_timing=False)
    select0_done = torch.npu.Event(enable_timing=False)
    select1_done = torch.npu.Event(enable_timing=False)
    gather0_done = torch.npu.Event(enable_timing=False)
    gather1_done = torch.npu.Event(enable_timing=False)
    tail_done = torch.npu.Event(enable_timing=False)
    outputs: dict[str, torch.Tensor] = {}

    start.record(torch.npu.current_stream())
    with torch.npu.stream(streams.compute):
        streams.compute.wait_event(start)
        outputs["indexer0"] = run_indexer_once(indexer_inputs)
        indexer0_done.record()
        outputs["indexer1"] = run_indexer_once(indexer_inputs)
        indexer1_done.record()

    with torch.npu.stream(streams.select):
        streams.select.wait_event(indexer0_done)
        run_mock_kv_select(half0, mock_wait_us)
        select0_done.record()
        streams.select.wait_event(indexer1_done)
        run_mock_kv_select(half1, mock_wait_us)
        select1_done.record()

    with torch.npu.stream(streams.gather):
        streams.gather.wait_event(select0_done)
        outputs["gather0_actual_seq"] = run_kv_gather_plan(half0)
        gather0_done.record()
        streams.gather.wait_event(select1_done)
        outputs["gather1_actual_seq"] = run_kv_gather_plan(half1)
        gather1_done.record()

    with torch.npu.stream(streams.compute):
        streams.compute.wait_event(select0_done)
        hit0 = run_hit_sfa(half0)
        streams.compute.wait_event(select1_done)
        hit1 = run_hit_sfa(half1)

        streams.compute.wait_event(gather0_done)
        outputs["attn0"] = run_miss_sfa_and_merge(half0, hit0)
        streams.compute.wait_event(gather1_done)
        outputs["attn1"] = run_miss_sfa_and_merge(half1, hit1)
        tail_done.record()

    torch.npu.current_stream().wait_event(tail_done)
    if synchronize:
        for stream in streams:
            stream.synchronize()
    return outputs


def run_pipeline_by_mode(
    *,
    pipeline_mode: str,
    streams: Streams | ThreeStreams,
    indexer_inputs,
    half0: HalfRunState,
    half1: HalfRunState,
    mock_wait_us: int,
    synchronize: bool = True,
) -> dict[str, torch.Tensor]:
    if pipeline_mode == "wide":
        return run_pipeline_once(
            streams=streams,
            indexer_inputs=indexer_inputs,
            half0=half0,
            half1=half1,
            mock_wait_us=mock_wait_us,
            synchronize=synchronize,
        )
    if pipeline_mode == "three":
        return run_pipeline_three_stream_once(
            streams=streams,
            indexer_inputs=indexer_inputs,
            half0=half0,
            half1=half1,
            mock_wait_us=mock_wait_us,
            synchronize=synchronize,
        )
    raise ValueError(f"unsupported pipeline_mode={pipeline_mode!r}")


def compare_outputs(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor],
                    *, atol: float, rtol: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in sorted(expected):
        exp = expected[name]
        act = actual[name]
        if exp.shape != act.shape or exp.dtype != act.dtype:
            rows.append(
                {
                    "name": name,
                    "ok": False,
                    "shape": f"{tuple(exp.shape)} vs {tuple(act.shape)}",
                    "dtype": f"{exp.dtype} vs {act.dtype}",
                    "max_abs": float("inf"),
                    "mean_abs": float("inf"),
                    "mismatch": -1,
                }
            )
            continue
        exp_cpu = exp.detach().cpu()
        act_cpu = act.detach().cpu()
        if exp_cpu.is_floating_point():
            diff = (exp_cpu.float() - act_cpu.float()).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
            ok = bool(torch.allclose(exp_cpu.float(), act_cpu.float(), atol=atol, rtol=rtol))
            mismatch = 0
        else:
            neq = exp_cpu != act_cpu
            mismatch = int(neq.sum().item())
            max_abs = float(mismatch > 0)
            mean_abs = max_abs
            ok = mismatch == 0
        rows.append(
            {
                "name": name,
                "ok": ok,
                "shape": tuple(exp.shape),
                "dtype": str(exp.dtype),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "mismatch": mismatch,
            }
        )
    return rows


def summarize(samples: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_avg_ms": float(samples.mean()),
        f"{prefix}_p50_ms": float(np.percentile(samples, 50)),
        f"{prefix}_p90_ms": float(np.percentile(samples, 90)),
        f"{prefix}_p99_ms": float(np.percentile(samples, 99)),
        f"{prefix}_min_ms": float(samples.min()),
        f"{prefix}_max_ms": float(samples.max()),
    }


def measure_serial_ms(
    *,
    stream: torch.npu.Stream,
    indexer_inputs,
    prepared0: PreparedHalf,
    prepared1: PreparedHalf,
    mock_wait_us: int,
    warmup: int,
    iters: int,
) -> np.ndarray:
    half0 = materialize_half(prepared0)
    half1 = materialize_half(prepared1)
    for _ in range(warmup):
        run_serial_once(
            stream=stream,
            indexer_inputs=indexer_inputs,
            half0=half0,
            half1=half1,
            mock_wait_us=mock_wait_us,
        )
    torch.npu.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        with torch.npu.stream(stream):
            start.record()
            run_serial_once(
                stream=stream,
                indexer_inputs=indexer_inputs,
                half0=half0,
                half1=half1,
                mock_wait_us=mock_wait_us,
            )
            end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return np.asarray(samples, dtype=np.float64)


def measure_pipeline_ms(
    *,
    pipeline_mode: str,
    streams: Streams | ThreeStreams,
    indexer_inputs,
    prepared0: PreparedHalf,
    prepared1: PreparedHalf,
    mock_wait_us: int,
    warmup: int,
    iters: int,
) -> np.ndarray:
    half0 = materialize_half(prepared0)
    half1 = materialize_half(prepared1)
    for _ in range(warmup):
        run_pipeline_by_mode(
            pipeline_mode=pipeline_mode,
            streams=streams,
            indexer_inputs=indexer_inputs,
            half0=half0,
            half1=half1,
            mock_wait_us=mock_wait_us,
        )
    torch.npu.synchronize()

    samples = []
    default_stream = torch.npu.current_stream()
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record(default_stream)
        run_pipeline_by_mode(
            pipeline_mode=pipeline_mode,
            streams=streams,
            indexer_inputs=indexer_inputs,
            half0=half0,
            half1=half1,
            mock_wait_us=mock_wait_us,
        )
        end.record(default_stream)
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return np.asarray(samples, dtype=np.float64)


def capture_pipeline_graph(
    *,
    pipeline_mode: str,
    streams: Streams | ThreeStreams,
    indexer_inputs,
    prepared0: PreparedHalf,
    prepared1: PreparedHalf,
    mock_wait_us: int,
    capture_warmup: int,
) -> CapturedPipelineGraph:
    half0 = materialize_half(prepared0)
    half1 = materialize_half(prepared1)
    for _ in range(capture_warmup):
        run_pipeline_by_mode(
            pipeline_mode=pipeline_mode,
            streams=streams,
            indexer_inputs=indexer_inputs,
            half0=half0,
            half1=half1,
            mock_wait_us=mock_wait_us,
        )
    torch.npu.synchronize()

    capture_stream = make_stream(torch.device(f"npu:{torch.npu.current_device()}"))
    graph = torch.npu.NPUGraph()
    holder: list[dict[str, torch.Tensor]] = []
    with torch.npu.graph(graph, stream=capture_stream, capture_error_mode="relaxed"):
        holder.append(
            run_pipeline_by_mode(
                pipeline_mode=pipeline_mode,
                streams=streams,
                indexer_inputs=indexer_inputs,
                half0=half0,
                half1=half1,
                mock_wait_us=mock_wait_us,
                synchronize=False,
            )
        )
    torch.npu.synchronize()
    return CapturedPipelineGraph(graph=graph, stream=capture_stream, outputs=holder[0])


def measure_graph_ms(
    captured: CapturedPipelineGraph,
    *,
    warmup: int,
    iters: int,
) -> np.ndarray:
    with torch.npu.stream(captured.stream):
        for _ in range(warmup):
            captured.graph.replay()
    torch.npu.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        with torch.npu.stream(captured.stream):
            start.record()
            captured.graph.replay()
            end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return np.asarray(samples, dtype=np.float64)


def make_profiler(trace_dir: Path, warmup: int, active: int):
    trace_dir.mkdir(parents=True, exist_ok=True)
    experimental_config = None
    try:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            export_type=torch_npu.profiler.ExportType.Text,
        )
    except Exception:
        pass
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir), analyse_flag=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental_config,
    )


def profile_pipeline(
    *,
    trace_dir: Path,
    pipeline_mode: str,
    streams: Streams | ThreeStreams,
    indexer_inputs,
    prepared0: PreparedHalf,
    prepared1: PreparedHalf,
    mock_wait_us: int,
    warmup: int,
    active: int,
) -> None:
    half0 = materialize_half(prepared0)
    half1 = materialize_half(prepared1)
    with make_profiler(trace_dir, warmup, active) as prof:
        for _ in range(warmup + active):
            run_pipeline_by_mode(
                pipeline_mode=pipeline_mode,
                streams=streams,
                indexer_inputs=indexer_inputs,
                half0=half0,
                half1=half1,
                mock_wait_us=mock_wait_us,
            )
            torch.npu.synchronize()
            prof.step()


def profile_graph(
    *,
    trace_dir: Path,
    captured: CapturedPipelineGraph,
    warmup: int,
    active: int,
) -> None:
    with make_profiler(trace_dir, warmup, active) as prof:
        for _ in range(warmup + active):
            with torch.npu.stream(captured.stream):
                captured.graph.replay()
            torch.npu.synchronize()
            prof.step()


def write_csv(row: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure split-batch MockKVSelect + KVGather + segmented SFA stream pipeline."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--full-batch-size", type=int, default=DEFAULT_FULL_BATCH)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--topk-reuse-rate", type=float, default=DEFAULT_REUSE_RATE)
    parser.add_argument("--mock-wait-us", type=int, default=DEFAULT_MOCK_WAIT_US)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--pipeline-mode", choices=("wide", "three"), default="wide")
    parser.add_argument("--offload-full-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gather-cube-cores", type=int, default=DEFAULT_GATHER_CUBE_CORES)
    parser.add_argument("--gather-vector-cores", type=int, default=DEFAULT_GATHER_VECTOR_CORES)
    parser.add_argument("--hit-sfa-cube-cores", type=int, default=DEFAULT_HIT_SFA_CUBE_CORES)
    parser.add_argument("--hit-sfa-vector-cores", type=int, default=DEFAULT_HIT_SFA_VECTOR_CORES)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-atol", type=float, default=1e-3)
    parser.add_argument("--validate-rtol", type=float, default=2e-2)
    parser.add_argument("--graph-replay", action="store_true")
    parser.add_argument("--graph-capture-warmup", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-active", type=int, default=3)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_batch_size % 2 != 0:
        raise SystemExit("--full-batch-size must be even")
    if not torch.npu.is_available():
        raise SystemExit("NPU not available")

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.set_device(device)
    print(f"device_index={_device_index(device)}", flush=True)

    half_batch = args.full_batch_size // 2
    print(
        "case: "
        f"full_bs={args.full_batch_size} half_bs={half_batch} max_seq={args.max_seq_len} "
        f"reuse={args.topk_reuse_rate} offload={args.offload_full_cache} "
        f"mock_wait_us={args.mock_wait_us} mode={args.pipeline_mode} "
        f"warmup={args.warmup} iters={args.iters}",
        flush=True,
    )

    print("preparing two half-batch KV/SFA plans...", flush=True)
    prepared0 = prepare_half(
        batch_size=half_batch,
        max_seq_len=args.max_seq_len,
        reuse_rate=args.topk_reuse_rate,
        device=device,
        seed=args.seed,
        offload_full_cache=args.offload_full_cache,
    )
    prepared1 = prepare_half(
        batch_size=half_batch,
        max_seq_len=args.max_seq_len,
        reuse_rate=args.topk_reuse_rate,
        device=device,
        seed=args.seed + 97,
        offload_full_cache=args.offload_full_cache,
    )
    print(
        f"half0 hit={prepared0.counts.hit} miss={prepared0.counts.miss}; "
        f"half1 hit={prepared1.counts.hit} miss={prepared1.counts.miss}",
        flush=True,
    )

    indexer_inputs = make_indexer_inputs(half_batch, args.max_seq_len, device)
    serial_stream = make_stream(device)
    streams = make_streams(device, args) if args.pipeline_mode == "wide" else make_three_streams(device, args)
    torch.npu.synchronize()

    if args.validate:
        print("validating serial vs pipeline outputs...", flush=True)
        serial_outputs = run_serial_once(
            stream=serial_stream,
            indexer_inputs=indexer_inputs,
            half0=materialize_half(prepared0),
            half1=materialize_half(prepared1),
            mock_wait_us=args.mock_wait_us,
        )
        pipeline_outputs = run_pipeline_by_mode(
            pipeline_mode=args.pipeline_mode,
            streams=streams,
            indexer_inputs=indexer_inputs,
            half0=materialize_half(prepared0),
            half1=materialize_half(prepared1),
            mock_wait_us=args.mock_wait_us,
        )
        rows = compare_outputs(
            serial_outputs,
            pipeline_outputs,
            atol=args.validate_atol,
            rtol=args.validate_rtol,
        )
        failed = [row for row in rows if not row["ok"]]
        for row in rows:
            status = "OK" if row["ok"] else "FAIL"
            detail = (
                f"mismatch={row['mismatch']}"
                if row["mismatch"]
                else f"max_abs={row['max_abs']:.6g} mean_abs={row['mean_abs']:.6g}"
            )
            print(f"  {status} {row['name']}: shape={row['shape']} dtype={row['dtype']} {detail}", flush=True)
        if failed:
            raise SystemExit(f"validation failed: {len(failed)} tensor(s) mismatched")
        print("validation passed", flush=True)

    print("measuring serial path...", flush=True)
    serial_ms = measure_serial_ms(
        stream=serial_stream,
        indexer_inputs=indexer_inputs,
        prepared0=prepared0,
        prepared1=prepared1,
        mock_wait_us=args.mock_wait_us,
        warmup=args.warmup,
        iters=args.iters,
    )
    print("measuring pipeline path...", flush=True)
    pipeline_ms = measure_pipeline_ms(
        pipeline_mode=args.pipeline_mode,
        streams=streams,
        indexer_inputs=indexer_inputs,
        prepared0=prepared0,
        prepared1=prepared1,
        mock_wait_us=args.mock_wait_us,
        warmup=args.warmup,
        iters=args.iters,
    )

    row: dict[str, object] = {
        "full_batch_size": args.full_batch_size,
        "half_batch_size": half_batch,
        "max_seq_len": args.max_seq_len,
        "reuse_rate": args.topk_reuse_rate,
        "offload_full_cache": args.offload_full_cache,
        "mock_wait_us": args.mock_wait_us,
        "pipeline_mode": args.pipeline_mode,
        "warmup": args.warmup,
        "iters": args.iters,
        "half0_hit": prepared0.counts.hit,
        "half0_miss": prepared0.counts.miss,
        "half1_hit": prepared1.counts.hit,
        "half1_miss": prepared1.counts.miss,
        "gather_cube_cores": args.gather_cube_cores,
        "gather_vector_cores": args.gather_vector_cores,
        "hit_sfa_cube_cores": args.hit_sfa_cube_cores,
        "hit_sfa_vector_cores": args.hit_sfa_vector_cores,
    }
    row.update(summarize(serial_ms, "serial"))
    row.update(summarize(pipeline_ms, "pipeline"))
    row["speedup_vs_serial"] = (
        row["serial_avg_ms"] / row["pipeline_avg_ms"]
        if row["pipeline_avg_ms"]
        else float("nan")
    )
    row["saved_ms"] = row["serial_avg_ms"] - row["pipeline_avg_ms"]
    write_csv(row, args.csv)

    print("\nsummary:")
    print(
        f"  serial:   {row['serial_avg_ms']:.3f}/{row['serial_p99_ms']:.3f} ms(avg/p99)",
        flush=True,
    )
    print(
        f"  pipeline: {row['pipeline_avg_ms']:.3f}/{row['pipeline_p99_ms']:.3f} ms(avg/p99)",
        flush=True,
    )
    print(
        f"  speedup={row['speedup_vs_serial']:.3f}x saved={row['saved_ms']:.3f} ms",
        flush=True,
    )

    captured = None
    if args.graph_replay:
        print("capturing graph replay path...", flush=True)
        captured = capture_pipeline_graph(
            pipeline_mode=args.pipeline_mode,
            streams=streams,
            indexer_inputs=indexer_inputs,
            prepared0=prepared0,
            prepared1=prepared1,
            mock_wait_us=args.mock_wait_us,
            capture_warmup=args.graph_capture_warmup,
        )
        captured.graph.replay()
        torch.npu.synchronize()
        if args.validate:
            graph_rows = compare_outputs(
                serial_outputs,
                captured.outputs,
                atol=args.validate_atol,
                rtol=args.validate_rtol,
            )
            failed = [graph_row for graph_row in graph_rows if not graph_row["ok"]]
            for graph_row in graph_rows:
                status = "OK" if graph_row["ok"] else "FAIL"
                detail = (
                    f"mismatch={graph_row['mismatch']}"
                    if graph_row["mismatch"]
                    else f"max_abs={graph_row['max_abs']:.6g} mean_abs={graph_row['mean_abs']:.6g}"
                )
                print(
                    f"  graph {status} {graph_row['name']}: "
                    f"shape={graph_row['shape']} dtype={graph_row['dtype']} {detail}",
                    flush=True,
                )
            if failed:
                raise SystemExit(f"graph validation failed: {len(failed)} tensor(s) mismatched")
            print("graph validation passed", flush=True)

        graph_ms = measure_graph_ms(captured, warmup=args.warmup, iters=args.iters)
        row.update(summarize(graph_ms, "graph"))
        row["graph_speedup_vs_serial"] = (
            row["serial_avg_ms"] / row["graph_avg_ms"]
            if row["graph_avg_ms"]
            else float("nan")
        )
        row["graph_speedup_vs_eager_pipeline"] = (
            row["pipeline_avg_ms"] / row["graph_avg_ms"]
            if row["graph_avg_ms"]
            else float("nan")
        )
        row["graph_saved_vs_pipeline_ms"] = row["pipeline_avg_ms"] - row["graph_avg_ms"]
        write_csv(row, args.csv)
        print(
            f"  graph:    {row['graph_avg_ms']:.3f}/{row['graph_p99_ms']:.3f} ms(avg/p99)",
            flush=True,
        )
        print(
            f"  graph speedup vs pipeline={row['graph_speedup_vs_eager_pipeline']:.3f}x "
            f"saved={row['graph_saved_vs_pipeline_ms']:.3f} ms",
            flush=True,
        )
    print(f"  csv: {args.csv}", flush=True)

    if args.profile:
        tag = (
            f"{args.pipeline_mode}{'_graph' if args.graph_replay else ''}_"
            f"bs{args.full_batch_size}_seq{args.max_seq_len}_reuse"
            f"{args.topk_reuse_rate:g}_wait{args.mock_wait_us}"
        ).replace(".", "p")
        trace_dir = args.trace_dir / tag
        print(f"profiling pipeline to {trace_dir}", flush=True)
        if args.graph_replay:
            if captured is None:
                captured = capture_pipeline_graph(
                    pipeline_mode=args.pipeline_mode,
                    streams=streams,
                    indexer_inputs=indexer_inputs,
                    prepared0=prepared0,
                    prepared1=prepared1,
                    mock_wait_us=args.mock_wait_us,
                    capture_warmup=args.graph_capture_warmup,
                )
            profile_graph(
                trace_dir=trace_dir,
                captured=captured,
                warmup=args.profile_warmup,
                active=args.profile_active,
            )
        else:
            profile_pipeline(
                trace_dir=trace_dir,
                pipeline_mode=args.pipeline_mode,
                streams=streams,
                indexer_inputs=indexer_inputs,
                prepared0=prepared0,
                prepared1=prepared1,
                mock_wait_us=args.mock_wait_us,
                warmup=args.profile_warmup,
                active=args.profile_active,
            )
        print(f"  profile: {trace_dir}", flush=True)


if __name__ == "__main__":
    main()
