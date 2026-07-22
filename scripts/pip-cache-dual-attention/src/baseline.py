import math
import random
from dataclasses import dataclass
from collections.abc import Callable
from typing import NamedTuple, TypeVar

import numpy as np
import torch
import torch_npu


@dataclass
class BaselineConfig:
    device: str = "npu:0"
    batch_size: int = 8
    seq_len: int = 1
    kv_max_seq_len: int = 8192
    block_size: int = 128
    index_topk: int = 2048
    seed: int = 2026
    indexer_num_heads: int = 64
    sparse_attn_num_heads: int = 128
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    indexer_head_dim: int = 128
    layout_query: str = "TND"
    layout_key: str = "PA_BSND"
    layout_kv: str = "PA_BSND"
    indexer_sparse_mode: int = 3
    sparse_attn_sparse_mode: int = 3
    sparse_attn_attention_mode: int = 2
    # 0: reinit block_status each gather step (cold pool); 1: keep status (pool hits after warm-up)
    topk_reuse_rate: float = 0.0

    @property
    def num_blocks(self) -> int:
        return max(1, (self.kv_max_seq_len + self.block_size - 1) // self.block_size)

    def validate(self) -> None:
        positive_int_fields = (
            "batch_size",
            "seq_len",
            "kv_max_seq_len",
            "block_size",
            "index_topk",
            "indexer_num_heads",
            "sparse_attn_num_heads",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "indexer_head_dim",
        )
        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        if not 0.0 <= self.topk_reuse_rate <= 1.0:
            raise ValueError(f"topk_reuse_rate must be in [0, 1], got {self.topk_reuse_rate}")
        # selection_topk_block_size=1: global ids are token indices on full KV (same as perf / DeepSeek decode).
        if self.index_topk > self.kv_max_seq_len:
            raise ValueError(
                f"index_topk ({self.index_topk}) must not exceed kv_max_seq_len ({self.kv_max_seq_len})"
            )
        if self.layout_query != "TND":
            raise ValueError(
                f"baseline requires layout_query='TND' (Gather topk/status use [B*S,H,TOPK]); got {self.layout_query!r}"
            )
        if self.layout_key != "PA_BSND":
            raise ValueError(f"unsupported layout_key: {self.layout_key}")
        if self.layout_kv != "PA_BSND":
            raise ValueError(f"unsupported layout_kv: {self.layout_kv}")


_T = TypeVar("_T")


@dataclass(frozen=True)
class StepMetrics:
    step_id: int
    indexer_ms: float
    gather_ms: float
    sparse_attn_ms: float
    step_ms: float


class IndexerInputs(NamedTuple):
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor
    block_table: torch.Tensor
    layout_query: str
    layout_key: str
    sparse_count: int
    sparse_mode: int


class GatherInputs(NamedTuple):
    selection_k_rope: torch.Tensor
    selection_kv_cache: torch.Tensor
    selection_kv_block_table: torch.Tensor
    selection_kv_block_status: torch.Tensor
    selection_topk_indices: torch.Tensor
    full_k_rope: torch.Tensor
    full_kv_cache: torch.Tensor
    full_kv_block_table: torch.Tensor
    full_kv_actual_seq: torch.Tensor
    full_q_actual_seq: torch.Tensor
    selection_topk_block_size: int


class SparseAttnInputs(NamedTuple):
    query: torch.Tensor
    query_rope: torch.Tensor
    sparse_indices: torch.Tensor
    scale_value: float
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    sparse_block_size: int
    layout_query: str
    layout_kv: str
    sparse_mode: int
    attention_mode: int


def alloc_tensor(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    *,
    fill_zero: bool,
    swapped: bool = False,
) -> torch.Tensor:
    try:
        tensor = (
            torch_npu.empty_with_swapped_memory(shape, dtype=dtype, device=device)
            if swapped
            else torch.empty(shape, dtype=dtype, device=device)
        )
    except Exception:
        tensor = torch.empty(shape, dtype=dtype, device=device)
    if fill_zero:
        tensor.zero_()
    elif dtype in (torch.float16, torch.float32, torch.bfloat16):
        tensor.normal_(0, 0.1)
    else:
        tensor.zero_()
    return tensor


def random_selection_topk(
    all_topk_nums: np.ndarray,
    token_count: int,
    head_num: int,
    topk: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """TND Gather layout: [token_count, head_num, topk] (token_count = B*S)."""
    if topk > len(all_topk_nums):
        raise ValueError(f"topk ({topk}) must not exceed available ids ({len(all_topk_nums)})")
    out = np.zeros((token_count, head_num, topk), dtype=np.int32)
    for t in range(token_count):
        for h in range(head_num):
            out[t, h] = rng.choice(all_topk_nums, size=topk, replace=False)
    return out


def reinit_selection_kv_block_status(selection_kv_block_status: torch.Tensor) -> None:
    """Clear cross-step reuse metadata (same as OffloadCache.reinit_status): all slots empty."""
    selection_kv_block_status.fill_(-1)


def blend_indexer_topk_with_reuse(
    indexer_topk: torch.Tensor,
    prev_topk: torch.Tensor | None,
    reuse_rate: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Host-side topk prep for Gather; excluded from ``run_step`` timing (not on NPU critical path)."""
    reuse_rate = max(0.0, min(1.0, reuse_rate))
    if prev_topk is None or reuse_rate <= 0.0:
        return indexer_topk
    if reuse_rate >= 1.0:
        return prev_topk.clone()
    topk = indexer_topk.shape[-1]
    n_keep = int(round(topk * reuse_rate))
    if n_keep <= 0:
        return indexer_topk
    if n_keep >= topk:
        return prev_topk.clone()
    cur = indexer_topk.detach().cpu().numpy()
    prev = prev_topk.detach().cpu().numpy()
    out = cur.copy()
    if out.ndim != 3:
        raise ValueError(f"blend expects TND topk [T,H,K], got shape {out.shape}")
    for t in range(out.shape[0]):
        for h in range(out.shape[1]):
            slots = rng.choice(topk, size=n_keep, replace=False)
            out[t, h, slots] = prev[t, h, slots]
    return torch.from_numpy(out).to(device=indexer_topk.device, dtype=indexer_topk.dtype)


def advance_topk_indices(
    topk: torch.Tensor,
    kv_max_seq_len: int,
    reuse_rate: float,
    rng: np.random.Generator,
) -> None:
    """In-place: keep ``reuse_rate`` fraction of slots; draw new global ids for the rest (matches perf)."""
    reuse_rate = max(0.0, min(1.0, reuse_rate))
    topk_count = topk.shape[-1]
    n_replace = int(round(topk_count * (1.0 - reuse_rate)))
    if n_replace <= 0:
        return
    all_ids = np.arange(0, kv_max_seq_len, dtype=np.int32)
    if n_replace >= topk_count:
        fresh = random_selection_topk(all_ids, topk.shape[0], topk.shape[1], topk_count, rng)
        topk.copy_(torch.from_numpy(fresh).to(device=topk.device, dtype=topk.dtype))
        return
    cur = topk.detach().cpu().numpy()
    out = cur.copy()
    for t in range(out.shape[0]):
        for h in range(out.shape[1]):
            slots = rng.choice(topk_count, size=n_replace, replace=False)
            kept = np.delete(out[t, h], slots)
            usable = np.setdiff1d(all_ids, kept, assume_unique=False)
            pick = min(n_replace, len(usable))
            if pick > 0:
                out[t, h, slots[:pick]] = rng.choice(usable, size=pick, replace=False)
    topk.copy_(torch.from_numpy(out).to(device=topk.device, dtype=topk.dtype))


def init_swapped_full_per_request(
    shape: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Swap-backed full cache: fill_(0) then host→device add_ one request (row slice) at a time."""
    n_rows = shape[0]
    rows_per_req, rem = divmod(n_rows, batch_size)
    if rem != 0:
        raise ValueError("full cache row count must be divisible by batch_size")
    out = torch_npu.empty_with_swapped_memory(shape, dtype=dtype, device=device)
    out.fill_(0)
    tail_shape = shape[1:]
    for b in range(batch_size):
        start = b * rows_per_req
        end = start + rows_per_req
        host = rng.uniform(size=(end - start, *tail_shape)).astype(np.float16)
        out[start:end].add_(torch.from_numpy(host).to(dtype).to(device=device))
    torch.npu.synchronize()
    return out


def prepare_gather_step(
    gather_inputs: GatherInputs,
    reuse_rate: float,
    kv_max_seq_len: int,
    rng: np.random.Generator,
) -> None:
    """reuse=0: clear pool (cold gather). reuse>0: roll topk like perf (replace 1-reuse fraction of slots)."""
    if reuse_rate <= 0.0:
        reinit_selection_kv_block_status(gather_inputs.selection_kv_block_status)
    else:
        advance_topk_indices(
            gather_inputs.selection_topk_indices, kv_max_seq_len, reuse_rate, rng
        )


class BaselineRuntime:
    def __init__(self, config: BaselineConfig | None = None):
        config = config or BaselineConfig()
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16
        self.topk_reuse_rate = config.topk_reuse_rate
        self.kv_max_seq_len = config.kv_max_seq_len
        self.indexer_sparse_mode = config.indexer_sparse_mode
        self.sparse_attn_sparse_mode = config.sparse_attn_sparse_mode
        self.sparse_attn_attention_mode = config.sparse_attn_attention_mode
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.block_size = config.block_size
        self.index_topk = config.index_topk
        self.num_blocks = config.num_blocks
        self.selection_topk_block_size = 1
        self.gather_head_num = 1
        self.selection_s_maxblocknum = max(
            1,
            (self.index_topk * self.selection_topk_block_size + self.block_size - 1) // self.block_size,
        )
        self.selection_num_blocks = (
            self.selection_s_maxblocknum * self.batch_size * self.seq_len * self.gather_head_num
        )
        self.indexer_num_heads = config.indexer_num_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.indexer_head_dim = config.indexer_head_dim
        self.query_layout = config.layout_query
        self.key_layout = config.layout_key
        self.kv_layout = config.layout_kv
        self.sparse_attn_num_heads = config.sparse_attn_num_heads
        self.token_count = self.batch_size * self.seq_len

        torch.npu.set_device(self.device)
        torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if hasattr(torch.npu, "manual_seed_all"):
            torch.npu.manual_seed_all(config.seed)

        try:
            import custom_ops  # noqa: F401
        except ImportError as exc:
            raise SystemExit("custom_ops required; build op/torch_ops_extension first.") from exc
        full_rows = (self.batch_size * self.num_blocks, self.block_size)
        self.gather_full_k_rope = init_swapped_full_per_request(
            (*full_rows, self.qk_rope_head_dim),
            self.batch_size,
            self.device,
            self.dtype,
            self.rng,
        )
        self.gather_full_kv_cache = init_swapped_full_per_request(
            (*full_rows, self.kv_lora_rank),
            self.batch_size,
            self.device,
            self.dtype,
            self.rng,
        )
        self.gather_full_block_table = torch.arange(
            self.batch_size * self.num_blocks,
            dtype=torch.int32,
            device=self.device,
        ).view(self.batch_size, self.num_blocks)
        self.indexer_kv_cache = alloc_tensor(
            (self.batch_size * self.num_blocks, self.block_size, 1, self.indexer_head_dim),
            self.device,
            self.dtype,
            fill_zero=False,
        )
        self.indexer_query_lengths = torch.full((self.batch_size,), self.seq_len, dtype=torch.int32, device=self.device)
        self.gather_kv_lengths = torch.full((self.batch_size,), config.kv_max_seq_len, dtype=torch.int32, device=self.device)

        if self.query_layout == "TND":
            token_count = self.batch_size * self.seq_len
            self.sparse_attn_query = torch.randn(
                (token_count, self.sparse_attn_num_heads, self.kv_lora_rank),
                dtype=self.dtype,
                device=self.device,
            )
            self.sparse_attn_query_rope = torch.randn(
                (token_count, self.sparse_attn_num_heads, self.qk_rope_head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            shape = (self.batch_size, self.seq_len)
            self.sparse_attn_query = torch.randn(
                (*shape, self.sparse_attn_num_heads, self.kv_lora_rank),
                dtype=self.dtype,
                device=self.device,
            )
            self.sparse_attn_query_rope = torch.randn(
                (*shape, self.sparse_attn_num_heads, self.qk_rope_head_dim),
                dtype=self.dtype,
                device=self.device,
            )

        self._indexer_inputs = self.make_indexer_inputs()
        self._gather_inputs = self.make_gather_inputs(self._initial_topk_indices())
        self._prev_topk: torch.Tensor | None = None

    def _initial_topk_indices(self) -> torch.Tensor:
        all_ids = np.arange(0, self.kv_max_seq_len, dtype=np.int32)
        topk_np = random_selection_topk(
            all_ids,
            self.token_count,
            self.gather_head_num,
            self.index_topk,
            self.rng,
        )
        return torch.from_numpy(topk_np).to(device=self.device, dtype=torch.int32)

    def make_indexer_inputs(self) -> IndexerInputs:
        """Match experiments/lightning_indexer/test_lightning_indexer_perf.make_inputs (TND)."""
        token_count = self.batch_size * self.seq_len
        return IndexerInputs(
            query=torch.randn(
                token_count, self.indexer_num_heads, self.indexer_head_dim, dtype=self.dtype, device=self.device
            ),
            key=self.indexer_kv_cache,
            weights=torch.randn(token_count, self.indexer_num_heads, dtype=self.dtype, device=self.device),
            actual_seq_lengths_query=torch.cumsum(self.indexer_query_lengths, dim=0).to(torch.int32),
            actual_seq_lengths_key=self.gather_kv_lengths,
            block_table=self.gather_full_block_table,
            layout_query=self.query_layout,
            layout_key=self.key_layout,
            sparse_count=self.index_topk,
            sparse_mode=self.indexer_sparse_mode,
        )

    def make_gather_inputs(self, selection_topk_indices: torch.Tensor | None = None) -> GatherInputs:
        batchseq = self.batch_size * self.seq_len
        selection_block_table = torch.arange(
            batchseq * self.selection_s_maxblocknum,
            dtype=torch.int32,
            device=self.device,
        ).reshape(batchseq, self.selection_s_maxblocknum)
        if selection_topk_indices is None:
            selection_topk_indices = self._initial_topk_indices()
        return GatherInputs(
            selection_k_rope=alloc_tensor(
                (self.selection_num_blocks, self.block_size, self.qk_rope_head_dim),
                self.device,
                self.dtype,
                fill_zero=True,
            ),
            selection_kv_cache=alloc_tensor(
                (self.selection_num_blocks, self.block_size, self.kv_lora_rank),
                self.device,
                self.dtype,
                fill_zero=True,
            ),
            selection_kv_block_table=selection_block_table,
            selection_kv_block_status=torch.full(
                (batchseq, self.gather_head_num, self.index_topk + 1),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            selection_topk_indices=selection_topk_indices,
            full_k_rope=self.gather_full_k_rope,
            full_kv_cache=self.gather_full_kv_cache,
            full_kv_block_table=self.gather_full_block_table,
            full_kv_actual_seq=self.gather_kv_lengths,
            full_q_actual_seq=self.indexer_query_lengths,
            selection_topk_block_size=self.selection_topk_block_size,
        )

    def make_sparse_attn_inputs(self, gather_kv_lengths: torch.Tensor) -> SparseAttnInputs:
        """Local column indices 0..topk-1 after gather (matches cann-recipes offload decode)."""
        default_topk = torch.arange(self.index_topk, dtype=torch.int32, device=self.device).view(1, 1, -1)
        default_topk = default_topk.expand(self.token_count, self.gather_head_num, -1)
        actual_seq = gather_kv_lengths.view(self.token_count)
        sparse_indices = torch.where(
            default_topk < actual_seq.view(self.token_count, 1, 1),
            default_topk,
            torch.full_like(default_topk, -1),
        )
        if self.query_layout == "TND":
            query = self.sparse_attn_query
            query_rope = self.sparse_attn_query_rope
            actual_seq_lengths_query = torch.cumsum(self.indexer_query_lengths, dim=0).to(torch.int32)
        else:
            sparse_indices = sparse_indices.view(
                self.batch_size, self.seq_len, self.gather_head_num, -1
            )
            query = self.sparse_attn_query
            query_rope = self.sparse_attn_query_rope
            actual_seq_lengths_query = self.indexer_query_lengths
        return SparseAttnInputs(
            query=query,
            query_rope=query_rope,
            sparse_indices=sparse_indices,
            scale_value=float(1.0 / math.sqrt(self.kv_lora_rank)),
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=gather_kv_lengths,
            sparse_block_size=self.selection_topk_block_size,
            layout_query=self.query_layout,
            layout_kv=self.kv_layout,
            sparse_mode=self.sparse_attn_sparse_mode,
            attention_mode=self.sparse_attn_attention_mode,
        )

    @staticmethod
    def _time_npu_ms(fn: Callable[[], _T]) -> tuple[_T, float]:
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        end.synchronize()
        return value, float(start.elapsed_time(end))

    def run_indexer(self, inputs: IndexerInputs | None = None) -> torch.Tensor:
        inputs = inputs or self._indexer_inputs
        output = torch_npu.npu_lightning_indexer(
            query=inputs.query,
            key=inputs.key,
            weights=inputs.weights,
            actual_seq_lengths_query=inputs.actual_seq_lengths_query,
            actual_seq_lengths_key=inputs.actual_seq_lengths_key,
            block_table=inputs.block_table,
            layout_query=inputs.layout_query,
            layout_key=inputs.layout_key,
            sparse_count=inputs.sparse_count,
            sparse_mode=inputs.sparse_mode,
        )
        return (output[0] if isinstance(output, (tuple, list)) else output).to(torch.int32)

    def run_gather(self, inputs: GatherInputs) -> torch.Tensor:
        return torch_npu.npu_gather_selection_kv_cache(
            selection_k_rope=inputs.selection_k_rope,
            selection_kv_cache=inputs.selection_kv_cache,
            selection_kv_block_table=inputs.selection_kv_block_table,
            selection_kv_block_status=inputs.selection_kv_block_status,
            selection_topk_indices=inputs.selection_topk_indices,
            full_k_rope=inputs.full_k_rope,
            full_kv_cache=inputs.full_kv_cache,
            full_kv_block_table=inputs.full_kv_block_table,
            full_kv_actual_seq=inputs.full_kv_actual_seq,
            full_q_actual_seq=inputs.full_q_actual_seq,
            selection_topk_block_size=inputs.selection_topk_block_size,
        ).to(torch.int32)

    def run_sparse_attn(
        self,
        gather_inputs: GatherInputs,
        sparse_attn_inputs: SparseAttnInputs,
    ) -> None:
        torch_npu.npu_sparse_flash_attention(
            query=sparse_attn_inputs.query,
            key=gather_inputs.selection_kv_cache.unsqueeze(2),
            value=gather_inputs.selection_kv_cache.unsqueeze(2),
            query_rope=sparse_attn_inputs.query_rope,
            key_rope=gather_inputs.selection_k_rope.unsqueeze(2),
            sparse_indices=sparse_attn_inputs.sparse_indices,
            scale_value=sparse_attn_inputs.scale_value,
            actual_seq_lengths_query=sparse_attn_inputs.actual_seq_lengths_query,
            actual_seq_lengths_kv=sparse_attn_inputs.actual_seq_lengths_kv,
            block_table=gather_inputs.selection_kv_block_table,
            sparse_block_size=sparse_attn_inputs.sparse_block_size,
            layout_query=sparse_attn_inputs.layout_query,
            layout_kv=sparse_attn_inputs.layout_kv,
            sparse_mode=sparse_attn_inputs.sparse_mode,
            attention_mode=sparse_attn_inputs.attention_mode,
        )

    def run_step(self, step_id: int) -> StepMetrics:
        torch.npu.synchronize()
        indexer_topk, indexer_ms = self._time_npu_ms(lambda: self.run_indexer())

        topk = blend_indexer_topk_with_reuse(
            indexer_topk, self._prev_topk, self.topk_reuse_rate, self.rng
        )

        gather_inputs = self._gather_inputs._replace(selection_topk_indices=topk)
        prepare_gather_step(
            gather_inputs, self.topk_reuse_rate, self.kv_max_seq_len, self.rng
        )
        gather_kv_lengths, gather_ms = self._time_npu_ms(lambda: self.run_gather(gather_inputs))
        self._prev_topk = topk.detach().clone()
        sparse_attn_inputs = self.make_sparse_attn_inputs(gather_kv_lengths)
        _, sparse_attn_ms = self._time_npu_ms(
            lambda: self.run_sparse_attn(gather_inputs, sparse_attn_inputs)
        )
        step_ms = indexer_ms + gather_ms + sparse_attn_ms
        return StepMetrics(
            step_id=step_id,
            indexer_ms=indexer_ms,
            gather_ms=gather_ms,
            sparse_attn_ms=sparse_attn_ms,
            step_ms=step_ms,
        )


def run_baseline_pipeline(config: BaselineConfig | None = None) -> list[StepMetrics]:
    runtime = BaselineRuntime(config)
    return [runtime.run_step(step_id) for step_id in range(runtime.seq_len)]


def _print_step(row: StepMetrics) -> None:
    print(
        f"step={row.step_id} indexer_ms={row.indexer_ms:.3f} gather_ms={row.gather_ms:.3f} "
        f"sparse_attn_ms={row.sparse_attn_ms:.3f} step_ms={row.step_ms:.3f}"
    )


if __name__ == "__main__":
    for row in run_baseline_pipeline():
        _print_step(row)
