"""Pure-Python reference implementations of the HBM index lookup / maintain ops.

These mirror the on-device kernels
``ascend-ops/asu_hbm_index_lookup/op_kernel/asu_hbm_index_lookup.cpp`` and
``ascend-ops/asu_hbm_index_maintain/op_kernel/asu_hbm_index_maintain.cpp`` so the
v0.1.1 compact SFA / v0.1 validate offload path can run during bring-up before the
real ``torch.ops._C_ascend.asu_hbm_index_*`` kernels are registered.

Only the two new HBM index ops are replaced; SFA and the lightning indexer still run
on their real (existing) kernels, so an NPU is still required end to end.

The numeric core (``hash32`` / ``lookup_request`` / ``maintain_request``) operates on
plain Python ``list[int]`` and has no torch dependency, so it is unit-testable on a
host without torch / NPU. The ``ref_hbm_index_lookup`` / ``ref_hbm_index_maintain``
wrappers adapt the torch tensor calling convention used by the manager and mutate the
state tensors in place, matching the real ops.
"""

NOT_FOUND = -1
_U32 = 0xFFFFFFFF


def hash32(x: int) -> int:
    """32-bit integer hash, identical to ``Hash32`` in the maintain kernel."""
    x &= _U32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & _U32
    x ^= x >> 15
    x = (x * 0x846CA68B) & _U32
    x ^= x >> 16
    return x & _U32


def lookup_request(
    index: list[int],
    slot_to_index: list[int],
    free_slots: list[int],
    free_head: int,
    query: list[int],
) -> tuple[list[int], int]:
    """Single-request lookup. Mutates ``index`` / ``slot_to_index`` in place.

    For each queried ``token_pos``: return its resident ``slot_id`` if already mapped,
    otherwise allocate the next free slot, wire the bidirectional map, and advance
    ``free_head``. Duplicate tokens within one call resolve to the same slot and
    consume a single free slot (the map is read fresh per token), matching the kernel.

    Returns ``(slot_out, new_free_head)``.
    """
    head = free_head
    slot_out: list[int] = []
    for token in query:
        slot = index[token]
        if slot == NOT_FOUND:
            slot = free_slots[head]
            head += 1
            index[token] = slot
            slot_to_index[slot] = token
        slot_out.append(slot)
    return slot_out, head


def maintain_request(
    index: list[int],
    slot_to_index: list[int],
    free_slots: list[int],
    free_head: int,
    last_query_slots: list[int],
    seed: int,
    req_id: int,
    slot_count: int,
) -> int:
    """Single-request maintain. Mutates ``index`` / ``slot_to_index`` / ``free_slots``.

    Reclaims occupied, unprotected slots back into the free pool until ``free_head``
    returns to 0, scanning circularly from a hashed start. Slots referenced by
    ``last_query_slots`` are protected (never evicted). Returns the new ``free_head``
    (always 0 when it started non-zero).
    """
    head = free_head
    if head == 0:
        return head
    protected = set(last_query_slots)
    slot = hash32((int(seed) ^ int(req_id)) & _U32) % slot_count
    steps_since_progress = 0
    while head > 0:
        index_id = slot_to_index[slot]
        if index_id != NOT_FOUND and slot not in protected:
            slot_to_index[slot] = NOT_FOUND
            index[index_id] = NOT_FOUND
            head -= 1
            free_slots[head] = slot
            steps_since_progress = 0
        else:
            steps_since_progress += 1
            if steps_since_progress > slot_count:
                raise ValueError(
                    "reference maintain cannot reclaim enough free slots: "
                    f"remaining free_head={head}, protected={len(protected)}, slot_count={slot_count}"
                )
        slot += 1
        if slot == slot_count:
            slot = 0
    return head


def ref_hbm_index_lookup(index, slot_to_index, free_slots, free_head, query_index, req_num):
    """Torch-tensor wrapper mirroring ``torch.ops._C_ascend.asu_hbm_index_lookup``.

    Mutates ``index`` / ``slot_to_index`` / ``free_head`` in place and returns a new
    ``slot_out`` tensor shaped like ``query_index`` (as the real op does).
    """
    import torch

    device = index.device
    dtype = index.dtype
    index_rows = index.detach().cpu().tolist()
    slot_to_index_rows = slot_to_index.detach().cpu().tolist()
    free_slots_rows = free_slots.detach().cpu().tolist()
    free_head_rows = free_head.detach().cpu().reshape(-1).tolist()
    query_rows = query_index.detach().cpu().tolist()

    slot_out_rows: list[list[int]] = []
    for req_id in range(int(req_num)):
        slot_out, new_head = lookup_request(
            index_rows[req_id],
            slot_to_index_rows[req_id],
            free_slots_rows[req_id],
            int(free_head_rows[req_id]),
            query_rows[req_id],
        )
        slot_out_rows.append(slot_out)
        free_head_rows[req_id] = new_head

    index.copy_(torch.tensor(index_rows, dtype=dtype, device=device))
    slot_to_index.copy_(torch.tensor(slot_to_index_rows, dtype=dtype, device=device))
    free_head.copy_(torch.tensor(free_head_rows, dtype=free_head.dtype, device=device).reshape(free_head.shape))
    return torch.tensor(slot_out_rows, dtype=query_index.dtype, device=device).reshape(query_index.shape)


def ref_hbm_index_maintain(index, slot_to_index, free_slots, free_head, last_query_slots, req_num, seed):
    """Torch-tensor wrapper mirroring ``torch.ops._C_ascend.asu_hbm_index_maintain_aicpu``.

    Mutates ``index`` / ``slot_to_index`` / ``free_slots`` / ``free_head`` in place.
    """
    import torch

    device = index.device
    slot_count = int(slot_to_index.shape[1])
    index_rows = index.detach().cpu().tolist()
    slot_to_index_rows = slot_to_index.detach().cpu().tolist()
    free_slots_rows = free_slots.detach().cpu().tolist()
    free_head_rows = free_head.detach().cpu().reshape(-1).tolist()
    last_query_slots_rows = last_query_slots.detach().cpu().tolist()

    for req_id in range(int(req_num)):
        new_head = maintain_request(
            index_rows[req_id],
            slot_to_index_rows[req_id],
            free_slots_rows[req_id],
            int(free_head_rows[req_id]),
            last_query_slots_rows[req_id],
            int(seed),
            req_id,
            slot_count,
        )
        free_head_rows[req_id] = new_head

    index.copy_(torch.tensor(index_rows, dtype=index.dtype, device=device))
    slot_to_index.copy_(torch.tensor(slot_to_index_rows, dtype=slot_to_index.dtype, device=device))
    free_slots.copy_(torch.tensor(free_slots_rows, dtype=free_slots.dtype, device=device))
    free_head.copy_(torch.tensor(free_head_rows, dtype=free_head.dtype, device=device).reshape(free_head.shape))
