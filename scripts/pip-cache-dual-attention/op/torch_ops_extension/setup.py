# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import multiprocessing
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, verify_ninja_availability

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
_CSRC_DIR = os.path.join(BASE_DIR, "custom_ops/csrc")

USE_NINJA = os.getenv('USE_NINJA') == '1'
MAX_JOBS = int(os.getenv('MAX_JOBS', multiprocessing.cpu_count()))

if USE_NINJA:
    verify_ninja_availability()

_gather_sources = [
    os.path.join(_CSRC_DIR, "npu_gather_selection_kv_cache.cpp"),
    os.path.join(_CSRC_DIR, "ops_def_registration_gather.cpp"),
]
_sfa_sources = [
    os.path.join(_CSRC_DIR, "npu_dmp_sparse_flash_attention.cpp"),
    os.path.join(_CSRC_DIR, "ops_def_registration_sparse.cpp"),
]
_merge_sources = [
    os.path.join(_CSRC_DIR, "npu_da_attention_merge.cpp"),
    os.path.join(_CSRC_DIR, "ops_def_registration_merge.cpp"),
]
_common_sources = [os.path.join(_CSRC_DIR, "ops_common.cpp")]
define_macros = []

if os.getenv("CUSTOM_OPS_SFA_ONLY") == "1":
    source_files = _sfa_sources + _merge_sources + _common_sources
    define_macros.append(("CUSTOM_OPS_SFA_ONLY", "1"))
elif os.getenv("CUSTOM_OPS_GATHER_ONLY") == "1":
    source_files = _gather_sources + _common_sources
    define_macros.append(("CUSTOM_OPS_GATHER_ONLY", "1"))
else:
    # Both ops: gather registration owns PYBIND11_MODULE; sparse adds TORCH_LIBRARY only.
    source_files = _gather_sources + _sfa_sources + _merge_sources + _common_sources

ext = NpuExtension(
    name="custom_ops.custom_ops_lib",
    sources=source_files,
    define_macros=define_macros,
    extra_compile_args=[
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        '-O3',
        '-march=native',
        '-ffast-math',
        '-fvisibility=hidden',
        '-flto',
    ],
    extra_link_args=['-flto'],
)

setup(
    name="custom_ops",
    version='1.0',
    keywords='custom_ops',
    ext_modules=[ext],
    package_data={
        'custom_ops': ['*.py', '*.so'],
        'custom_ops.converter': ['*.py', '*.so'],
    },
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA, parallel=MAX_JOBS)},
)
