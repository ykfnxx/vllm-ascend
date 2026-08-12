# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

set(dsa_sparse_host_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/dsa_sparse_lookup_update_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dsa_sparse_lookup_update_infershape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dsa_sparse_lookup_update_tiling.cpp
)

npu_op_code_gen(
    SRC ${dsa_sparse_host_sources}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -I${PROJECT_SOURCE_DIR}/include
)

# The production op_api wrapper is copied under op_host/op_api for source
# parity. vllm-ascend builds it against generated aclnnInner* symbols. The
# standard standalone npu_op_code_gen path emits the public aclnn* entry points
# directly, so compiling that wrapper here would duplicate the public symbols.
file(GLOB autogen_aclnn_sources
    ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp
)
file(GLOB autogen_proto_sources
    ${ASCEND_AUTOGEN_PATH}/op_proto.cc
    ${ASCEND_AUTOGEN_PATH}/group_op_proto/*.cc
)
file(GLOB autogen_fallback_sources
    ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp
)
set_source_files_properties(
    ${autogen_aclnn_sources}
    ${autogen_proto_sources}
    ${autogen_fallback_sources}
    PROPERTIES GENERATED TRUE
)

npu_op_library(cust_opapi ACLNN
    ${autogen_aclnn_sources}
)
npu_op_library(cust_op_proto GRAPH
    ${dsa_sparse_host_sources}
    ${autogen_proto_sources}
)
npu_op_library(cust_optiling TILING
    ${dsa_sparse_host_sources}
    ${autogen_fallback_sources}
)

foreach(target cust_op_proto cust_optiling)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_definitions(${target} PRIVATE
        LOG_CPP
        OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
        OPS_UTILS_LOG_PACKAGE_TYPE=""
    )
    target_compile_options(${target} PRIVATE
        -fvisibility=hidden
        -fno-strict-aliasing
    )
endforeach()
target_compile_options(cust_opapi PRIVATE -fvisibility=hidden)

npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)
