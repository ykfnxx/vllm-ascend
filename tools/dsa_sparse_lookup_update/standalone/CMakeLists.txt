# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

cmake_minimum_required(VERSION 3.19.0)
project(dsa_sparse_lookup_update_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH OR
   ASCEND_CANN_PACKAGE_PATH STREQUAL "")
    if(DEFINED ENV{ASCEND_HOME_PATH} AND
       NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann")
    endif()
endif()

find_package(
    ASC REQUIRED
    HINTS
        ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake
)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR
   ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ascend950)
endif()
if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    message(FATAL_ERROR
        "DsaSparseLookupUpdate SIMT supports only ascend950; got "
        "${ASCEND_COMPUTE_UNIT}")
endif()

set(package_name dsa_sparse_prof)
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        ENABLE_SOURCE_PACKAGE True
        ENABLE_BINARY_PACKAGE True
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
