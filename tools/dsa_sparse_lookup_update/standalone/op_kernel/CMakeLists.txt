# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

# MindStudio msOpProf requires debug information for source/PC mapping. Keep
# the production optimization level and add only the profiling information.
npu_op_kernel_options(
    ascendc_kernels
    DsaSparseLookupUpdate
    COMPUTE_UNIT ascend950
    OPTIONS
        -g
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE DsaSparseLookupUpdate
    COMPUTE_UNIT ascend950
    KERNEL_DIR .
    KERNEL_FILE dsa_sparse_lookup_update.cpp
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
