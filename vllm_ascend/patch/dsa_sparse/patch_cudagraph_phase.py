"""Add the DSA row phase to the vLLM v0.18 cudagraph key."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace

import vllm.forward_context as forward_context
import vllm.v1.cudagraph_dispatcher as dispatcher_mod
from vllm.config import CUDAGraphMode

_BaseBatchDescriptor = forward_context.BatchDescriptor


@dataclass(frozen=True)
class BatchDescriptor(_BaseBatchDescriptor):
    dsa_graph_phase: str | None = None


def _dispatch(
    self,
    num_tokens: int,
    uniform_decode: bool = False,
    has_lora: bool = False,
    num_active_loras: int = 0,
    dsa_graph_phase: str | None = None,
    valid_modes: AbstractSet[CUDAGraphMode] | None = None,
    invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
) -> tuple[CUDAGraphMode, BatchDescriptor]:
    allowed_modes = valid_modes or CUDAGraphMode.valid_runtime_modes()
    if invalid_modes:
        allowed_modes -= invalid_modes
    assert allowed_modes

    if (
        not self.keys_initialized
        or self.cudagraph_mode == CUDAGraphMode.NONE
        or num_tokens > self.compilation_config.max_cudagraph_capture_size
        or allowed_modes <= {CUDAGraphMode.NONE}
    ):
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

    effective_num_active_loras = num_active_loras
    if has_lora and num_active_loras > 0:
        if self.specialize_lora_count:
            import bisect

            index = bisect.bisect_left(
                self.captured_lora_counts, num_active_loras
            )
            if index < len(self.captured_lora_counts):
                effective_num_active_loras = self.captured_lora_counts[index]
        else:
            assert self.vllm_config.lora_config is not None
            effective_num_active_loras = (
                self.vllm_config.lora_config.max_loras + 1
            )

    batch_desc = self._create_padded_batch_descriptor(
        num_tokens,
        uniform_decode and self.cudagraph_mode.separate_routine(),
        has_lora,
        effective_num_active_loras,
    )
    batch_desc = replace(batch_desc, dsa_graph_phase=dsa_graph_phase)
    if (
        CUDAGraphMode.FULL in allowed_modes
        and batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]
    ):
        return CUDAGraphMode.FULL, batch_desc

    piecewise_desc = replace(batch_desc, num_reqs=None, uniform=False)
    if (
        CUDAGraphMode.PIECEWISE in allowed_modes
        and piecewise_desc in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]
    ):
        return CUDAGraphMode.PIECEWISE, piecewise_desc

    assert CUDAGraphMode.NONE in allowed_modes
    return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)


forward_context.BatchDescriptor = BatchDescriptor
dispatcher_mod.BatchDescriptor = BatchDescriptor
dispatcher_mod.CudagraphDispatcher.dispatch = _dispatch
