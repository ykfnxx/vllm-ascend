#!/usr/bin/env python3
"""Smoke test for MockKVSelect through the handwritten aclnn AICPU path."""

from __future__ import annotations

import argparse
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import acl


ACL_SUCCESS = 0
ACL_FLOAT16 = 1
ACL_INT32 = 3
ACL_FORMAT_ND = 2
ACL_MEM_MALLOC_NORMAL_ONLY = 0


def check(ret: int, what: str) -> None:
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{what} failed: {ret}")


def elem_count(shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def dtype_size(dtype: int) -> int:
    if dtype == ACL_FLOAT16:
        return 2
    if dtype == ACL_INT32:
        return 4
    raise ValueError(f"unsupported dtype in smoke test: {dtype}")


def contiguous_strides(shape: list[int]) -> list[int]:
    strides = [1] * len(shape)
    for idx in range(len(shape) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * shape[idx + 1]
    return strides


class AclnnLibs:
    def __init__(self, opapi_path: Path) -> None:
        mode = os.RTLD_GLOBAL | os.RTLD_NOW
        self.nnopbase = ctypes.CDLL("libnnopbase.so", mode=mode)
        self.opapi = ctypes.CDLL(str(opapi_path), mode=mode)

        self.nnopbase.aclCreateTensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.nnopbase.aclCreateTensor.restype = ctypes.c_void_p
        self.nnopbase.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        self.nnopbase.aclDestroyTensor.restype = ctypes.c_int

        tensor_args = [ctypes.c_void_p] * 10
        output_args = [ctypes.c_void_p] * 8
        self.opapi.aclnnMockKVSelectGetWorkspaceSize.argtypes = [
            *tensor_args,
            ctypes.c_int64,
            ctypes.c_int64,
            *output_args,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.opapi.aclnnMockKVSelectGetWorkspaceSize.restype = ctypes.c_int
        self.opapi.aclnnMockKVSelect.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.opapi.aclnnMockKVSelect.restype = ctypes.c_int


@dataclass
class AclTensor:
    ptr: int
    tensor: int
    shape_buf: ctypes.Array[ctypes.c_int64]
    stride_buf: ctypes.Array[ctypes.c_int64]
    storage_shape_buf: ctypes.Array[ctypes.c_int64]
    libs: AclnnLibs

    @classmethod
    def create(cls, libs: AclnnLibs, dtype: int, shape: list[int]) -> "AclTensor":
        size = elem_count(shape) * dtype_size(dtype)
        ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check(ret, "acl.rt.malloc")

        shape_buf = (ctypes.c_int64 * len(shape))(*shape)
        strides = contiguous_strides(shape)
        stride_buf = (ctypes.c_int64 * len(strides))(*strides)
        storage_shape_buf = (ctypes.c_int64 * len(shape))(*shape)
        tensor = libs.nnopbase.aclCreateTensor(
            shape_buf,
            len(shape),
            dtype,
            stride_buf,
            0,
            ACL_FORMAT_ND,
            storage_shape_buf,
            len(shape),
            ctypes.c_void_p(int(ptr)),
        )
        if not tensor:
            check(acl.rt.free(ptr), "acl.rt.free")
            raise RuntimeError("aclCreateTensor failed")
        return cls(
            ptr=int(ptr),
            tensor=int(tensor),
            shape_buf=shape_buf,
            stride_buf=stride_buf,
            storage_shape_buf=storage_shape_buf,
            libs=libs,
        )

    def destroy(self) -> None:
        check(self.libs.nnopbase.aclDestroyTensor(ctypes.c_void_p(self.tensor)), "aclDestroyTensor")
        check(acl.rt.free(self.ptr), "acl.rt.free")


def make_specs(batch_size: int, seq_len: int, head_num: int, head_dim: int,
               max_seq_len: int, topk: int, block_size: int) -> tuple[list[tuple[int, list[int]]],
                                                                       list[tuple[int, list[int]]]]:
    block_count = max(1, (max_seq_len + block_size - 1) // block_size)
    token_shape = [batch_size, seq_len, head_num, head_dim]
    cache_shape = [batch_size, block_count, head_num, head_dim]
    block_table_shape = [batch_size, seq_len, head_num, block_count]
    topk_shape = [batch_size, seq_len, head_num, topk]
    row_shape = [batch_size, seq_len, head_num]

    input_specs = [
        (ACL_FLOAT16, token_shape),       # selection_k_rope
        (ACL_FLOAT16, cache_shape),       # selection_kv_cache
        (ACL_INT32, block_table_shape),   # selection_kv_block_table
        (ACL_INT32, block_table_shape),   # selection_kv_block_status
        (ACL_INT32, topk_shape),          # selection_topk_indices
        (ACL_FLOAT16, token_shape),       # full_k_rope
        (ACL_FLOAT16, cache_shape),       # full_kv_cache
        (ACL_INT32, block_table_shape),   # full_kv_block_table
        (ACL_INT32, row_shape),           # full_kv_actual_seq
        (ACL_INT32, row_shape),           # full_q_actual_seq
    ]
    output_specs = [
        (ACL_INT32, topk_shape),          # hit_sparse_indices
        (ACL_INT32, topk_shape),          # miss_topk_indices
        (ACL_INT32, topk_shape),          # miss_insert_indices
        (ACL_INT32, row_shape),           # hit_actual_seq
        (ACL_INT32, row_shape),           # miss_actual_seq
        (ACL_INT32, row_shape),           # miss_count
        (ACL_INT32, row_shape),           # hit_count
        (ACL_INT32, row_shape),           # selection_status_empty
    ]
    return input_specs, output_specs


def default_opapi_path() -> Path:
    project_dir = Path(__file__).resolve().parents[1]
    return project_dir / "build/output/vendors/customize_asn/op_api/lib/libcust_opapi.so"


def run_case(args: argparse.Namespace) -> None:
    libs = AclnnLibs(args.opapi)
    check(acl.init(), "acl.init")
    check(acl.rt.set_device(args.device_id), "acl.rt.set_device")
    stream, ret = acl.rt.create_stream()
    check(ret, "acl.rt.create_stream")

    inputs: list[AclTensor] = []
    outputs: list[AclTensor] = []
    workspace_ptr: int | None = None
    try:
        input_specs, output_specs = make_specs(
            args.batch_size,
            args.seq_len,
            args.head_num,
            args.head_dim,
            args.max_seq_len,
            args.topk,
            args.block_size,
        )
        inputs = [AclTensor.create(libs, dtype, shape) for dtype, shape in input_specs]
        outputs = [AclTensor.create(libs, dtype, shape) for dtype, shape in output_specs]

        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = libs.opapi.aclnnMockKVSelectGetWorkspaceSize(
            *[ctypes.c_void_p(tensor.tensor) for tensor in inputs],
            ctypes.c_int64(args.block_size),
            ctypes.c_int64(args.mock_wait_us),
            *[ctypes.c_void_p(tensor.tensor) for tensor in outputs],
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        check(ret, "aclnnMockKVSelectGetWorkspaceSize")

        workspace_arg = ctypes.c_void_p()
        if workspace_size.value > 0:
            workspace_ptr, ret = acl.rt.malloc(workspace_size.value, ACL_MEM_MALLOC_NORMAL_ONLY)
            check(ret, "acl.rt.malloc(workspace)")
            workspace_arg = ctypes.c_void_p(int(workspace_ptr))

        ret = libs.opapi.aclnnMockKVSelect(
            workspace_arg,
            workspace_size,
            executor,
            ctypes.c_void_p(int(stream)),
        )
        check(ret, "aclnnMockKVSelect")
        check(acl.rt.synchronize_stream(stream), "acl.rt.synchronize_stream")
        print(f"MockKVSelect ACLNN AICPU smoke test passed. workspace={workspace_size.value}")
    finally:
        if workspace_ptr is not None:
            check(acl.rt.free(workspace_ptr), "acl.rt.free(workspace)")
        for tensor in inputs + outputs:
            tensor.destroy()
        check(acl.rt.destroy_stream(stream), "acl.rt.destroy_stream")
        check(acl.rt.reset_device(args.device_id), "acl.rt.reset_device")
        check(acl.finalize(), "acl.finalize")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MockKVSelect AICPU ACLNN smoke test.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--head-num", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--mock-wait-us", type=int, default=25)
    parser.add_argument("--opapi", type=Path, default=default_opapi_path())
    args = parser.parse_args()
    run_case(args)


if __name__ == "__main__":
    main()
