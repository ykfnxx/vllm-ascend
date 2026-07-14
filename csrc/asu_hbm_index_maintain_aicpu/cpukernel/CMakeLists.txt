cmake_minimum_required(VERSION 3.16)

project(asu_hbm_index_maintain_aicpu_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(AICPU_BUILTIN_ROOT
    ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel)
set(AICPU_BUILTIN_LIB_DIR ${AICPU_BUILTIN_ROOT}/lib)

add_library(cust_aicpu_kernels SHARED
    impl/asu_hbm_index_maintain_aicpu_kernel.cc
)

target_compile_definitions(cust_aicpu_kernels PRIVATE
    _GLIBCXX_USE_CXX11_ABI=1
)

target_include_directories(cust_aicpu_kernels PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include/aicpu
)

target_link_libraries(cust_aicpu_kernels PRIVATE
    -Wl,--whole-archive
    ${AICPU_BUILTIN_LIB_DIR}/libascend_protobuf.a
    ${AICPU_BUILTIN_LIB_DIR}/libcpu_kernels_context.a
    -Wl,--no-whole-archive
)

target_link_options(cust_aicpu_kernels PRIVATE
    -s
    -Wl,-Bsymbolic
    -Wl,--exclude-libs=libascend_protobuf.a
)
