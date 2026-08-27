# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest


class SpyIOBackend:
    def __init__(self) -> None:
        self.put_calls = []
        self.get_calls = []

    def put_blocks(self, **kwargs) -> None:
        self.put_calls.append(
            {key: value.clone() if hasattr(value, "clone") else value for key, value in kwargs.items()}
        )

    def get_tokens(self, **kwargs) -> None:
        self.get_calls.append(
            {key: value.clone() if hasattr(value, "clone") else value for key, value in kwargs.items()}
        )


@pytest.fixture
def spy_io() -> SpyIOBackend:
    return SpyIOBackend()
