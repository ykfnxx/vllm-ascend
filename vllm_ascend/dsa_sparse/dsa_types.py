# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DSA 稀疏卸载跨模块共享的轻量类型与常量。

本文件只放轻量、稳定、可被 scheduler/worker/算子边界共同引用的类型。
INVALID_SLOT 是 scheduler/worker 之间传递 resident 状态时使用的哨兵值；
DSASparseRole 描述 manager 当前运行在 scheduler 侧还是 worker 侧；
ReqStage 描述请求生命周期里的 DSA 阶段；DSADecodeRowMode 描述传给
gather-selection/SFA 的每行执行模式。

不要在这里引入重型运行时依赖，避免基础类型模块反向耦合具体实现。
"""

import enum


INVALID_SLOT = -1


class DSASparseRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1


class DSADecodeRowMode(enum.IntEnum):
    """Per-row execution mode for DSA decode operator boundaries.

    ReqStage is scheduler-owned request state.  This enum is the tensorized row
    contract consumed by gather-selection/SFA style decode operators:
    - PAD rows are graph padding and must not touch cache state.
    - DENSE rows keep the native full-cache attention indices.
    - SPARSE rows materialize selected DRAM tokens into resident sparse budget
      slots and use resident-cache logical attention indices.
    """

    PAD = 0
    DENSE = 1
    SPARSE = 2


class ReqStage(enum.IntEnum):
    """DSA sparse-cache stage for one request in one scheduler step.

    This is the scheduler-owned state machine used by both allocation and the
    worker runtime. It deliberately separates "what cache layout this request
    uses now" from layer-local actions such as dumping a newly completed block.

    Transitions:
    - PREFILL -> DENSE_DECODE after prompt/chunk prefill is done but the full
      context is still below the DSA sparse threshold, or sparse decode is not
      supported for the current step.
    - DENSE_DECODE/PREFILL -> ENTER_SPARSE_DECODE on the first decode step
      whose context can use DSA sparse MLA/SFA. This includes both the classic
      long-prompt first decode and the short-prompt long-decode case where the
      request crosses the threshold later.
    - ENTER_SPARSE_DECODE -> SPARSE_DECODE on the next sparse decode step.

    Full-block dump is an action that may happen in PREFILL, DENSE_DECODE, or
    SPARSE_DECODE when a block becomes complete. It is not a separate stage.
    """

    PREFILL = 0
    DENSE_DECODE = 1
    ENTER_SPARSE_DECODE = 2
    SPARSE_DECODE = 3

    @classmethod
    def coerce(cls, value: object) -> "ReqStage":
        if isinstance(value, cls):
            return value
        try:
            return cls(int(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return cls.PREFILL

    @property
    def is_decode(self) -> bool:
        return self != ReqStage.PREFILL

    @property
    def is_sparse_decode(self) -> bool:
        return self in (ReqStage.ENTER_SPARSE_DECODE, ReqStage.SPARSE_DECODE)

    @property
    def is_enter_sparse_decode(self) -> bool:
        return self == ReqStage.ENTER_SPARSE_DECODE
