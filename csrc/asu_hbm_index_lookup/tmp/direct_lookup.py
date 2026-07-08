import ctypes
from functools import lru_cache
from pathlib import Path
from typing import Callable


def _current_stream_ptr(torch_module) -> int:
    return int(torch_module.npu.current_stream().npu_stream)


@lru_cache(maxsize=None)
def _load_lookup_function(library_path: str):
    library = ctypes.CDLL(
        str(Path(library_path).expanduser().resolve()),
        mode=ctypes.RTLD_GLOBAL,
    )
    function = library.asu_hbm_index_lookup_do
    function.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    function.restype = None
    return function


def load_direct_lookup_op(library_path: str, block_dim: int = 1) -> Callable:
    function = _load_lookup_function(library_path)

    def direct_lookup(
        index,
        slot_to_index,
        free_slots,
        free_head,
        query_index,
        req_num,
    ):
        import torch

        slot_out = torch.empty_like(query_index)
        function(
            ctypes.c_uint32(block_dim),
            ctypes.c_void_p(_current_stream_ptr(torch)),
            ctypes.c_void_p(index.data_ptr()),
            ctypes.c_void_p(slot_to_index.data_ptr()),
            ctypes.c_void_p(free_slots.data_ptr()),
            ctypes.c_void_p(free_head.data_ptr()),
            ctypes.c_void_p(query_index.data_ptr()),
            ctypes.c_void_p(slot_out.data_ptr()),
            ctypes.c_uint32(req_num),
        )
        return slot_out

    return direct_lookup
