import ctypes
from functools import lru_cache
from pathlib import Path
from typing import Callable


def _current_stream_ptr(torch_module) -> int:
    return int(torch_module.npu.current_stream().npu_stream)


@lru_cache(maxsize=None)
def _load_maintain_function(library_path: str):
    library = ctypes.CDLL(
        str(Path(library_path).expanduser().resolve()),
        mode=ctypes.RTLD_GLOBAL,
    )
    function = library.asu_hbm_index_maintain_do
    function.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    function.restype = None
    return function


def load_direct_maintain_op(library_path: str, block_dim: int = 1) -> Callable:
    function = _load_maintain_function(library_path)

    def direct_maintain(
        index,
        slot_to_index,
        free_slots,
        free_head,
        last_query_slots,
        req_num,
        seed,
    ) -> None:
        import torch

        function(
            ctypes.c_uint32(block_dim),
            ctypes.c_void_p(_current_stream_ptr(torch)),
            ctypes.c_void_p(index.data_ptr()),
            ctypes.c_void_p(slot_to_index.data_ptr()),
            ctypes.c_void_p(free_slots.data_ptr()),
            ctypes.c_void_p(free_head.data_ptr()),
            ctypes.c_void_p(last_query_slots.data_ptr()),
            ctypes.c_uint32(req_num),
            ctypes.c_uint32(seed),
        )

    return direct_maintain
