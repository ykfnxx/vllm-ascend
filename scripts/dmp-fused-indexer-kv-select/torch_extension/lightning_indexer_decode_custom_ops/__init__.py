import importlib
import os
from pathlib import Path

import torch
import torch_npu


def _prepend_custom_opp_path(path: Path) -> None:
    lib_path = path / "op_api" / "lib" / "libcust_opapi.so"
    if not lib_path.exists():
        return
    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    parts = [item for item in current.split(":") if item]
    text = str(path)
    if text not in parts:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([text] + parts)


def _read_recorded_install_opp() -> str:
    env_file = Path(__file__).resolve().parents[2] / ".lightning_indexer_decode_env"
    if not env_file.exists():
        return ""
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    key = "LIGHTNING_INDEXER_DECODE_INSTALL_OPP_PATH="
    for line in lines:
        if line.startswith(key):
            return line[len(key):].strip()
    return ""


def _setup_custom_opp_path() -> None:
    install_opp = os.environ.get("LIGHTNING_INDEXER_DECODE_INSTALL_OPP_PATH") or _read_recorded_install_opp()
    if install_opp:
        _prepend_custom_opp_path(Path(install_opp) / "vendors" / "customize")

    ascend_opp = os.environ.get("ASCEND_OPP_PATH")
    if ascend_opp:
        _prepend_custom_opp_path(Path(ascend_opp) / "vendors" / "customize")

    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        _prepend_custom_opp_path(Path(ascend_home) / "opp" / "vendors" / "customize")


_setup_custom_opp_path()
custom_ops_lib = importlib.import_module(".custom_ops_lib", __name__)

_custom = getattr(torch.ops, "custom", None)
if _custom is not None and hasattr(_custom, "npu_lightning_indexer_decode"):
    _decode_op = getattr(_custom, "npu_lightning_indexer_decode")

    def npu_lightning_indexer_decode(
        query,
        key,
        weights,
        *,
        actual_seq_lengths_key,
        block_table,
    ):
        return _decode_op(query, key, weights, actual_seq_lengths_key, block_table)

    setattr(torch_npu, "npu_lightning_indexer_decode", npu_lightning_indexer_decode)

if _custom is not None and hasattr(_custom, "npu_lightning_indexer_decode_update"):
    _decode_update_op = getattr(_custom, "npu_lightning_indexer_decode_update")

    def npu_lightning_indexer_decode_update(
        query,
        key,
        weights,
        cache_slots,
        *,
        actual_seq_lengths_key,
        block_table,
    ):
        return _decode_update_op(query, key, weights, cache_slots, actual_seq_lengths_key, block_table)

    setattr(torch_npu, "npu_lightning_indexer_decode_update", npu_lightning_indexer_decode_update)

__all__ = ["custom_ops_lib"]
if "npu_lightning_indexer_decode" in globals():
    __all__.append("npu_lightning_indexer_decode")
if "npu_lightning_indexer_decode_update" in globals():
    __all__.append("npu_lightning_indexer_decode_update")
