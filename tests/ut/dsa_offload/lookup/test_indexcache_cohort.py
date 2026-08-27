# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest

from vllm_ascend.dsa_offload.lookup import scan_index_cache_cohorts


def test_glm_indexcache_cohorts_follow_execution_order() -> None:
    cohorts = scan_index_cache_cohorts(
        (
            ("layer.0", False, 0),
            ("layer.1", True, 1),
            ("layer.2", True, 2),
            ("layer.3", False, 3),
        )
    )

    assert [cohort.cohort_id for cohort in cohorts] == ["layer.0", "layer.3"]
    assert cohorts[0].layer_names == ("layer.0", "layer.1", "layer.2")
    assert cohorts[0].layer_ids == (0, 1, 2)


def test_follower_without_leader_fails() -> None:
    with pytest.raises(ValueError, match="must follow a cohort leader"):
        scan_index_cache_cohorts((("layer.0", True, 0),))

    with pytest.raises(ValueError, match="requires target SFA layers"):
        scan_index_cache_cohorts(())


def test_follower_must_be_consecutive_and_layer_order_stable() -> None:
    with pytest.raises(ValueError, match="followers must be consecutive"):
        scan_index_cache_cohorts(
            (
                ("layer.0", False, 0),
                ("layer.2", True, 2),
            )
        )

    with pytest.raises(ValueError, match="must follow execution order"):
        scan_index_cache_cohorts(
            (
                ("layer.1", False, 1),
                ("layer.0", False, 0),
            )
        )
