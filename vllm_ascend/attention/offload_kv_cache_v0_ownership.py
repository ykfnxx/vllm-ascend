from collections.abc import Iterable
from dataclasses import dataclass


KV_PAYLOAD_DOMAIN = 0
INDEXER_KEY_DOMAIN = 1

NORMAL_KV_BLOCK = 0
OFFLOAD_KV_BLOCK = 1
INDEXER_BLOCK = 2


@dataclass
class BlockOwnershipRegistry:
    total_blocks: int
    offload_blocks: Iterable[int]

    def __post_init__(self) -> None:
        self._owners = [
            [NORMAL_KV_BLOCK for _ in range(self.total_blocks)],
            [INDEXER_BLOCK for _ in range(self.total_blocks)],
        ]
        self._offload_blocks = [int(block_id) for block_id in self.offload_blocks]
        for block_id in self._offload_blocks:
            if block_id < 0 or block_id >= self.total_blocks:
                raise ValueError(f"offload block id {block_id} is outside total_blocks={self.total_blocks}")
            self._owners[KV_PAYLOAD_DOMAIN][block_id] = OFFLOAD_KV_BLOCK

    def owner(self, storage_domain: int, block_id: int) -> int:
        return self._owners[int(storage_domain)][int(block_id)]

    def normal_kv_blocks(self) -> list[int]:
        return [
            block_id
            for block_id, owner in enumerate(self._owners[KV_PAYLOAD_DOMAIN])
            if owner == NORMAL_KV_BLOCK
        ]

    def offload_kv_blocks(self) -> list[int]:
        return list(self._offload_blocks)

    def assert_original_kv_blocks(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            if self.owner(KV_PAYLOAD_DOMAIN, int(block_id)) != NORMAL_KV_BLOCK:
                raise ValueError(f"original K/V metadata references offload block {int(block_id)}")

    def assert_indexer_blocks(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            if self.owner(INDEXER_KEY_DOMAIN, int(block_id)) != INDEXER_BLOCK:
                raise ValueError(f"indexer metadata references non-indexer block {int(block_id)}")

    def assert_compact_kv_blocks(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            if self.owner(KV_PAYLOAD_DOMAIN, int(block_id)) != OFFLOAD_KV_BLOCK:
                raise ValueError(f"compact metadata references non-offload K/V block {int(block_id)}")


def compact_blocks_per_req(slot_count: int, block_size: int) -> int:
    if slot_count <= 0:
        raise ValueError(f"slot_count must be positive, got {slot_count}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return (slot_count + block_size - 1) // block_size


def offload_reserved_blocks(max_pinned_reqs: int, blocks_per_req: int) -> int:
    reserved_blocks = int(max_pinned_reqs) * int(blocks_per_req)
    if reserved_blocks <= 0:
        raise ValueError(
            f"offload reserved blocks must be positive, got max_pinned_reqs={max_pinned_reqs}, "
            f"blocks_per_req={blocks_per_req}"
        )
    return reserved_blocks


def offload_reserved_bytes(reserved_blocks: int, page_size_bytes_total: int) -> int:
    if reserved_blocks <= 0:
        raise ValueError(f"reserved_blocks must be positive, got {reserved_blocks}")
    if page_size_bytes_total <= 0:
        raise ValueError(f"page_size_bytes_total must be positive, got {page_size_bytes_total}")
    return int(reserved_blocks) * int(page_size_bytes_total)


def inflated_tensor_size(size_bytes: int, page_size_bytes: int, reserved_blocks: int) -> int:
    if size_bytes <= 0:
        raise ValueError(f"size_bytes must be positive, got {size_bytes}")
    if page_size_bytes <= 0:
        raise ValueError(f"page_size_bytes must be positive, got {page_size_bytes}")
    if size_bytes % page_size_bytes != 0:
        raise ValueError(
            f"tensor size {size_bytes} is not a multiple of page_size_bytes {page_size_bytes}"
        )
    return int(size_bytes) + offload_reserved_bytes(reserved_blocks, page_size_bytes)


def build_static_offload_blocks(total_blocks: int, max_pinned_reqs: int, blocks_per_req: int) -> list[int]:
    reserved_blocks = offload_reserved_blocks(max_pinned_reqs, blocks_per_req)
    if reserved_blocks >= total_blocks:
        raise ValueError(
            "offload reserved blocks must leave at least one normal K/V block: "
            f"reserved={reserved_blocks}, total={total_blocks}"
        )
    return list(range(total_blocks - reserved_blocks, total_blocks))


def build_compact_block_table_row(
    registry: BlockOwnershipRegistry,
    offload_blocks: Iterable[int],
) -> list[int]:
    row = [int(block_id) for block_id in offload_blocks]
    registry.assert_compact_kv_blocks(row)
    return row


def physical_slot_for_compact_slot(
    slot_id: int,
    block_size: int,
    compact_block_table_row: list[int],
) -> int:
    logical_block = int(slot_id) // int(block_size)
    block_offset = int(slot_id) % int(block_size)
    if logical_block < 0 or logical_block >= len(compact_block_table_row):
        raise ValueError(
            f"compact slot {slot_id} is outside compact block table with {len(compact_block_table_row)} blocks"
        )
    return int(compact_block_table_row[logical_block]) * int(block_size) + block_offset
