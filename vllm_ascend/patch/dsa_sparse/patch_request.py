"""Attach the DSA cache-layout state to vLLM v0.18 requests."""

from functools import wraps

from vllm.v1.request import Request
from vllm_ascend.dsa_sparse.dsa_types import INVALID_SLOT, ReqStage

_original_init = Request.__init__


@wraps(_original_init)
def _request_init(self: Request, *args, **kwargs) -> None:
    _original_init(self, *args, **kwargs)
    self.dsa_req_stage = ReqStage.PREFILL
    self.dsa_next_req_stage = ReqStage.PREFILL
    self.dsa_resident_valid_seq_len = INVALID_SLOT
    self.dsa_sparse_budget_tokens = 0


Request.__init__ = _request_init
