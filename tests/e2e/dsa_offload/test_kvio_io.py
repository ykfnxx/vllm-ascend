# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.dsa_offload.io import make_storage_ids
from vllm_ascend.dsa_offload.kvio import KVIOBackend
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = [
    pytest.mark.ascend_a5,
    pytest.mark.skipif(
        get_ascend_device_type() != AscendDeviceType.A5,
        reason="KVIO DSA Offload IO requires Ascend A5",
    ),
]


def test_real_block_put_and_discrete_token_get() -> None:
    block_size = 4
    source = torch.arange(2 * block_size * 8, dtype=torch.float16, device="npu").reshape(2, block_size, 8)
    destination = torch.zeros_like(source)
    backend = KVIOBackend(1)
    backend.register_put_cache(layer_id=0, block_size=block_size, cache_planes=(source,))
    backend.register_get_cache(layer_id=0, block_size=block_size, cache_planes=(destination,))
    backend.finalize_registration()

    storage_ids = make_storage_ids([bytes(range(32))], 0, device=source.device)
    backend.put_blocks(
        layer_id=0,
        storage_ids=storage_ids,
        source_block_ids=torch.tensor([1], dtype=torch.int64, device=source.device),
    )
    backend.get_tokens(
        layer_id=0,
        storage_ids=storage_ids.repeat(2),
        token_offsets=torch.tensor([0, 3], dtype=torch.int64, device=source.device),
        destination_slots=torch.tensor([0, 1], dtype=torch.int64, device=source.device),
    )

    assert torch.equal(destination.reshape(-1, 8)[0], source[1, 0])
    assert torch.equal(destination.reshape(-1, 8)[1], source[1, 3])
    backend.close()
