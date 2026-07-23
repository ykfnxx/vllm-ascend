from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace


EXAMPLES_DIR = (
    Path(__file__).resolve().parents[3] / "examples" / "dsa_sparse"
)
sys.path.insert(0, str(EXAMPLES_DIR))
SCRIPT_PATH = EXAMPLES_DIR / "compare_asu_hbm_index_maintain_modes.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_asu_hbm_index_maintain_modes", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
COMPARISON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARISON
SPEC.loader.exec_module(COMPARISON)


def make_args(**overrides):
    values = {
        "batch_size": 32,
        "miss_count": 300,
        "capture_warmup_iterations": 3,
        "warmup_iterations": 10,
        "iterations": 100,
        "profile_iterations": 20,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_args_accepts_default_comparison_shape() -> None:
    COMPARISON.validate_args(make_args())


def test_validate_args_rejects_invalid_iteration_count() -> None:
    try:
        COMPARISON.validate_args(make_args(iterations=0))
    except ValueError as exc:
        assert str(exc) == "--iterations must be greater than 0"
    else:
        raise AssertionError("expected invalid iteration count to fail")


def test_summarize_mode_preserves_samples_and_mode() -> None:
    result = COMPARISON.summarize_mode(
        "graph", batch_size=32, miss_count=300, samples_ms=[0.1, 0.2]
    )

    assert result["mode"] == "graph"
    assert result["operator"] == "maintain_graph"
    assert math.isclose(result["mean_ms"], 0.15)
    assert result["samples_ms"] == [0.1, 0.2]
