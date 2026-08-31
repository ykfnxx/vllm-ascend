# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        dsa_sparse_turbo_resolve_update_batch_v2_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME DsaSparseTurboResolveUpdateBatchV2
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE dsa_sparse_turbo_resolve_update_batch_v2
        ACLNNTYPE aclnn_inner
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
