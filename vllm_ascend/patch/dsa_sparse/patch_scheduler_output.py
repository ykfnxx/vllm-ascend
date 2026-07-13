"""Extend vLLM v0.18 scheduler messages with DSA state."""

from dataclasses import dataclass, field
from typing import Any

import vllm.v1.core.sched.output as output_mod

_BaseNewRequestData = output_mod.NewRequestData
_BaseCachedRequestData = output_mod.CachedRequestData
_BaseSchedulerOutput = output_mod.SchedulerOutput


@dataclass
class NewRequestData(_BaseNewRequestData):
    block_hashes: list[Any] | None = None


@dataclass
class CachedRequestData(_BaseCachedRequestData):
    block_hashes: list[list[Any]] = field(default_factory=list)

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
            block_hashes=[],
        )


@dataclass
class SchedulerOutput(_BaseSchedulerOutput):
    req_dsa_stage: dict[str, int] | None = None
    req_dsa_resident_valid_seq_len: dict[str, int] | None = None
    req_dsa_sparse_budget_tokens: dict[str, int] | None = None

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        output = super().make_empty()
        output.req_dsa_stage = {}
        output.req_dsa_resident_valid_seq_len = {}
        output.req_dsa_sparse_budget_tokens = {}
        return output


output_mod.NewRequestData = NewRequestData
output_mod.CachedRequestData = CachedRequestData
output_mod.SchedulerOutput = SchedulerOutput
