# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlockHashUpdate:
    base_count: int
    hashes: tuple[bytes, ...]
    replace: bool = False


def apply_block_hash_update(
    request_id: str,
    committed_hashes: list[bytes],
    update: BlockHashUpdate,
) -> None:
    if update.replace:
        committed_hashes[:] = update.hashes
        return

    if len(committed_hashes) < update.base_count:
        raise RuntimeError(
            "DSA Offload block hash update has a gap for request "
            f"{request_id}: local={len(committed_hashes)}, "
            f"base={update.base_count}."
        )

    overlap = min(
        len(committed_hashes) - update.base_count,
        len(update.hashes),
    )
    if any(
        committed_hashes[update.base_count + index] != update.hashes[index]
        for index in range(overlap)
    ):
        raise RuntimeError(
            "DSA Offload block hashes diverged from the scheduler "
            f"for request {request_id}."
        )
    committed_hashes.extend(update.hashes[overlap:])
