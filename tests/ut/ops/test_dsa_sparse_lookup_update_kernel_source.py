# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from pathlib import Path


KERNEL_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "csrc"
    / "attention"
    / "dsa_sparse_lookup_update"
    / "op_kernel"
    / "arch35"
    / "dsa_sparse_lookup_update_simt.h"
)


def test_outputs_are_initialized_before_inactive_request_return() -> None:
    source = KERNEL_SOURCE.read_text(encoding="utf-8")

    inactive_return = source.index("if (!has_active_query)")
    resolved_initialization = source.index(
        "resolved_hot_indices[output_offset] ="
    )
    miss_initialization = source.index("miss_mask[output_offset] = 0U")

    assert resolved_initialization < inactive_return
    assert miss_initialization < inactive_return
