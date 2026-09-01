# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import hashlib
from dataclasses import dataclass
from typing import Any, TypeAlias

DecodeHashContext: TypeAlias = tuple[
    int,
    bytes | None,
    tuple[int, ...],
    tuple[Any, ...] | None,
]
CommittedBlockUpdate: TypeAlias = tuple[int, tuple[int, ...]]

_BLOCK_KEY_DOMAIN = b"dsa-offload-block-key-v1"
_INT63_MASK = (1 << 63) - 1


def make_block_key(canonical_hash: bytes) -> int:
    digest = hashlib.sha256(
        _BLOCK_KEY_DOMAIN + canonical_hash
    ).digest()
    block_key = int.from_bytes(digest[:8], "big") & _INT63_MASK
    return block_key or 1


def apply_committed_update(
    request_id: str,
    committed: list[int],
    update: CommittedBlockUpdate,
) -> None:
    base_count, appended = update
    if len(committed) < base_count:
        raise RuntimeError(
            "DSA Offload block-key update has a gap for request "
            f"{request_id}: local={len(committed)}, base={base_count}."
        )
    overlap = min(len(committed) - base_count, len(appended))
    if committed[base_count : base_count + overlap] != list(
        appended[:overlap]
    ):
        raise RuntimeError(
            "DSA Offload block keys diverged from the scheduler for request "
            f"{request_id}."
        )
    committed.extend(appended[overlap:])


@dataclass(frozen=True, slots=True)
class DSAOffloadStepMetadata:
    committed_updates: dict[str, CommittedBlockUpdate]
    decode_contexts: dict[str, DecodeHashContext]
    candidate_keys: dict[str, tuple[int, ...]]
