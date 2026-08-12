# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

cmake_minimum_required(VERSION 3.16.0)
project(dsa_sparse_lookup_update_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_HOME_PATH OR ASCEND_HOME_PATH STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH must name the CANN installation")
endif()
if(NOT DEFINED CUSTOM_OP_ROOT OR CUSTOM_OP_ROOT STREQUAL "")
    message(FATAL_ERROR "CUSTOM_OP_ROOT must name the installed vendor root")
endif()

find_library(ASCENDCL_LIBRARY NAMES ascendcl
    PATHS
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/x86_64-linux/lib64
        ${ASCEND_HOME_PATH}/aarch64-linux/lib64
    NO_DEFAULT_PATH
)
find_library(NNOPBASE_LIBRARY NAMES nnopbase
    PATHS
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/x86_64-linux/lib64
        ${ASCEND_HOME_PATH}/aarch64-linux/lib64
    NO_DEFAULT_PATH
)
find_library(OPAPI_LIBRARY NAMES opapi
    PATHS
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/x86_64-linux/lib64
        ${ASCEND_HOME_PATH}/aarch64-linux/lib64
    NO_DEFAULT_PATH
)
find_library(CUSTOM_OP_LIBRARY NAMES cust_opapi
    PATHS ${CUSTOM_OP_ROOT}/op_api/lib
    NO_DEFAULT_PATH
)
foreach(library
        ASCENDCL_LIBRARY NNOPBASE_LIBRARY OPAPI_LIBRARY CUSTOM_OP_LIBRARY)
    if(NOT ${library})
        message(FATAL_ERROR "Required library ${library} was not found")
    endif()
endforeach()

add_executable(
    dsa_sparse_lookup_update_runner
    dsa_sparse_lookup_update_runner.cpp
)
target_include_directories(dsa_sparse_lookup_update_runner PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/x86_64-linux/include
    ${ASCEND_HOME_PATH}/aarch64-linux/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_host/op_api
)
target_link_libraries(dsa_sparse_lookup_update_runner PRIVATE
    ${ASCENDCL_LIBRARY}
    ${NNOPBASE_LIBRARY}
    ${OPAPI_LIBRARY}
    ${CUSTOM_OP_LIBRARY}
)
get_filename_component(CUSTOM_OP_LIBRARY_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
get_filename_component(ASCENDCL_LIBRARY_DIR ${ASCENDCL_LIBRARY} DIRECTORY)
set_target_properties(dsa_sparse_lookup_update_runner PROPERTIES
    BUILD_RPATH "${CUSTOM_OP_LIBRARY_DIR};${ASCENDCL_LIBRARY_DIR}"
)
