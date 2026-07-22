cmake_minimum_required(VERSION 3.16)
project(mock_kv_select_aicpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
    elseif(EXISTS "/usr/local/Ascend/cann-8.5.1")
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-8.5.1")
    else()
        message(FATAL_ERROR "Set ASCEND_CANN_PACKAGE_PATH or ASCEND_HOME_PATH.")
    endif()
endif()

if(NOT DEFINED ASCEND_PYTHON_EXECUTABLE)
    set(ASCEND_PYTHON_EXECUTABLE python3)
endif()

if(NOT DEFINED VENDOR_NAME)
    set(VENDOR_NAME customize_asn)
endif()

set(AICPU_INC_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc/aicpu/cpu_kernels")
if(NOT EXISTS "${AICPU_INC_DIR}/cpu_kernel.h")
    set(AICPU_INC_DIR "${ASCEND_CANN_PACKAGE_PATH}/include/aicpu")
endif()
set(AICPU_PUBLIC_INC_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/aicpu")
set(AICPU_LIB_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64")
set(AICPU_CONTEXT_LIB "${AICPU_LIB_DIR}/libaicpu_context.a")
set(AICPU_PROTOBUF_LIB "${AICPU_LIB_DIR}/libbase_ascend_protobuf.a")
set(AICPU_SEC_LIB "${AICPU_LIB_DIR}/libc_sec.a")
set(AICPU_ARM_CXX "${ASCEND_CANN_PACKAGE_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++")
set(AICPU_JSON_SRC "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/mock_kv_select.json")
set(AICPU_JSON "${CMAKE_CURRENT_BINARY_DIR}/cust_aicpu_kernel.json")
set(AICPU_KERNEL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/mock_kv_select_kernels.o")
set(AICPU_KERNEL_SO_NAME libasn_aicpu_kernels.so)
set(AICPU_KERNEL_SO "${CMAKE_CURRENT_BINARY_DIR}/${AICPU_KERNEL_SO_NAME}")

if(NOT EXISTS "${AICPU_INC_DIR}/cpu_kernel.h")
    message(FATAL_ERROR "AICPU headers not found: ${AICPU_INC_DIR}")
endif()
if(NOT EXISTS "${AICPU_CONTEXT_LIB}")
    message(FATAL_ERROR "AICPU context library not found: ${AICPU_CONTEXT_LIB}")
endif()
if(NOT EXISTS "${AICPU_PROTOBUF_LIB}")
    message(FATAL_ERROR "AICPU protobuf library not found: ${AICPU_PROTOBUF_LIB}")
endif()
if(NOT EXISTS "${AICPU_SEC_LIB}")
    message(FATAL_ERROR "AICPU securec library not found: ${AICPU_SEC_LIB}")
endif()
if(NOT EXISTS "${AICPU_ARM_CXX}")
    message(FATAL_ERROR "AICPU aarch64 compiler not found: ${AICPU_ARM_CXX}")
endif()
if(NOT EXISTS "${AICPU_JSON_SRC}")
    message(FATAL_ERROR "AICPU json not found: ${AICPU_JSON_SRC}")
endif()

add_custom_command(
    OUTPUT "${AICPU_KERNEL_OBJ}"
    COMMAND "${AICPU_ARM_CXX}" -std=c++11 -fPIC -fvisibility=hidden
            "-I${AICPU_INC_DIR}"
            "-I${AICPU_PUBLIC_INC_DIR}"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/cpukernel/impl/mock_kv_select_kernels.cc"
            -o "${AICPU_KERNEL_OBJ}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpukernel/impl/mock_kv_select_kernels.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/cpukernel/impl/mock_kv_select_kernels.h"
    COMMENT "Compile MockKVSelect AICPU kernel"
    VERBATIM
)

add_custom_command(
    OUTPUT "${AICPU_KERNEL_SO}"
    COMMAND "${AICPU_ARM_CXX}" -shared "${AICPU_KERNEL_OBJ}"
            -Wl,--whole-archive "${AICPU_CONTEXT_LIB}" "${AICPU_PROTOBUF_LIB}" "${AICPU_SEC_LIB}"
            -Wl,--no-whole-archive
            -Wl,-Bsymbolic
            -Wl,--exclude-libs=libbase_ascend_protobuf.a:libc_sec.a
            -Wl,-z,now
            -s
            -o "${AICPU_KERNEL_SO}"
    DEPENDS "${AICPU_KERNEL_OBJ}"
    COMMENT "Link ${AICPU_KERNEL_SO_NAME}"
    VERBATIM
)
add_custom_target(aicpu_kernels ALL DEPENDS "${AICPU_KERNEL_SO}")

add_custom_command(
    OUTPUT "${AICPU_JSON}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${AICPU_JSON_SRC}" "${AICPU_JSON}"
    DEPENDS "${AICPU_JSON_SRC}"
    COMMENT "Package cust_aicpu_kernel.json"
    VERBATIM
)
add_custom_target(aicpu_op_info ALL DEPENDS "${AICPU_JSON}")

set(OUTPUT_ROOT "${CMAKE_CURRENT_BINARY_DIR}/output/vendors/${VENDOR_NAME}")
add_custom_target(package_aicpu ALL
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${OUTPUT_ROOT}/op_impl/cpu/aicpu_kernel/impl"
            "${OUTPUT_ROOT}/op_impl/cpu/config"
    COMMAND "${CMAKE_COMMAND}" -E copy "${AICPU_KERNEL_SO}"
            "${OUTPUT_ROOT}/op_impl/cpu/aicpu_kernel/impl/${AICPU_KERNEL_SO_NAME}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${AICPU_JSON}"
            "${OUTPUT_ROOT}/op_impl/cpu/config/cust_aicpu_kernel.json"
    DEPENDS aicpu_kernels aicpu_op_info
)

install(FILES "${AICPU_KERNEL_SO}"
    DESTINATION "vendors/${VENDOR_NAME}/op_impl/cpu/aicpu_kernel/impl")
install(FILES "${AICPU_JSON}"
    DESTINATION "vendors/${VENDOR_NAME}/op_impl/cpu/config")

option(BUILD_MOCK_KV_SELECT_OPAPI "Build handwritten aclnn op_api for MockKVSelect AICPU launch" ON)
if(BUILD_MOCK_KV_SELECT_OPAPI)
    add_library(cust_opapi SHARED op_api/mock_kv_select_aclnn.cpp)
    set_target_properties(cust_opapi PROPERTIES OUTPUT_NAME cust_opapi)
    target_compile_features(cust_opapi PRIVATE cxx_std_17)
    target_compile_definitions(cust_opapi PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(cust_opapi PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/op_api"
        "${ASCEND_CANN_PACKAGE_PATH}/include"
        "${ASCEND_CANN_PACKAGE_PATH}/include/aclnn"
        "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc"
        "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include"
        "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/aclnn")
    target_link_directories(cust_opapi PRIVATE
        "${ASCEND_CANN_PACKAGE_PATH}/lib64"
        "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64")
    target_link_libraries(cust_opapi PRIVATE
        nnopbase
        profapi
        ge_common_base
        ascend_dump
        ascendalog
        dl)
    add_custom_target(package_opapi ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${OUTPUT_ROOT}/op_api/lib"
                "${OUTPUT_ROOT}/op_api/include"
        COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:cust_opapi>"
                "${OUTPUT_ROOT}/op_api/lib/libcust_opapi.so"
        COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/op_api/mock_kv_select_aclnn.h"
                "${OUTPUT_ROOT}/op_api/include/aclnn_mock_kv_select.h"
        DEPENDS cust_opapi
    )
    add_dependencies(package_aicpu package_opapi)
    install(TARGETS cust_opapi
        LIBRARY DESTINATION "vendors/${VENDOR_NAME}/op_api/lib")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/op_api/mock_kv_select_aclnn.h"
        DESTINATION "vendors/${VENDOR_NAME}/op_api/include"
        RENAME "aclnn_mock_kv_select.h")
endif()

option(BUILD_MOCK_KV_SELECT_PROTO "Build shape infer proto shared library for compile checking" OFF)
if(BUILD_MOCK_KV_SELECT_PROTO)
    add_library(cust_opsproto_mock_kv_select SHARED op_proto/mock_kv_select_proto.cpp)
    target_compile_definitions(cust_opsproto_mock_kv_select PRIVATE OP_PROTO_LIB)
    target_include_directories(cust_opsproto_mock_kv_select PRIVATE
        "${ASCEND_CANN_PACKAGE_PATH}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../ascendc/src/utils/inc")
    target_link_directories(cust_opsproto_mock_kv_select PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
    target_link_libraries(cust_opsproto_mock_kv_select PRIVATE
        exe_graph graph graph_base register ascendalog error_manager platform c_sec)
    add_custom_command(TARGET cust_opsproto_mock_kv_select POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${OUTPUT_ROOT}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}"
        COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:cust_opsproto_mock_kv_select>"
                "${OUTPUT_ROOT}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}/libcust_opsproto_mock_kv_select.so"
        VERBATIM)
endif()

option(BUILD_MOCK_KV_SELECT_TF_PLUGIN "Build TensorFlow parser plugin that marks MockKVSelect as AI_CPU" ON)
if(BUILD_MOCK_KV_SELECT_TF_PLUGIN)
    add_library(cust_tf_parsers SHARED framework/tf_plugin/mock_kv_select_tf_plugin.cpp)
    target_include_directories(cust_tf_parsers PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
    target_link_directories(cust_tf_parsers PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
    target_link_libraries(cust_tf_parsers PRIVATE graph register)
    add_custom_target(package_framework ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${OUTPUT_ROOT}/framework/tensorflow"
        COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:cust_tf_parsers>"
                "${OUTPUT_ROOT}/framework/tensorflow/libcust_tf_parsers.so"
        DEPENDS cust_tf_parsers
    )
    add_dependencies(package_aicpu package_framework)
    install(TARGETS cust_tf_parsers
        LIBRARY DESTINATION "vendors/${VENDOR_NAME}/framework/tensorflow")
endif()
