# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.dsa_offload.io import MockIOBackend, create_io_backend


def test_mock_backend_is_a_pure_noop() -> None:
    backend = MockIOBackend()
    cache = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    original = cache.clone()
    values = torch.tensor([1, 2], dtype=torch.int64)

    backend.register_put_cache(layer_id=0, block_size=2, cache_planes=(cache,))
    backend.register_get_cache(layer_id=0, block_size=2, cache_planes=(cache,))
    backend.finalize_registration()
    backend.put_blocks(layer_id=0, storage_ids=values, source_block_ids=values)
    backend.get_tokens(
        layer_id=0,
        storage_ids=values,
        token_offsets=values,
        destination_slots=values,
    )
    backend.close()

    assert torch.equal(cache, original)
    assert vars(backend) == {}


def test_factory_selects_only_explicit_backends() -> None:
    assert isinstance(create_io_backend("mock", 9), MockIOBackend)

    kvio = create_io_backend("kvio", 9)
    assert kvio.__class__.__name__ == "KVIOBackend"

    with pytest.raises(ValueError, match="Unsupported DSA Offload IO backend"):
        create_io_backend("memory", 9)
