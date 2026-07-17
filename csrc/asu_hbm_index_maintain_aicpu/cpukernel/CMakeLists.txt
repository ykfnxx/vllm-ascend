cmake_minimum_required(VERSION 3.16)

project(asu_hbm_index_maintain_aicpu_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(AICPU_LIB_DIR ${ASCEND_CANN_PACKAGE_PATH}/lib64)

add_library(cust_aicpu_kernels SHARED
    impl/asu_hbm_index_maintain_aicpu_kernel.cc
)
set_target_properties(cust_aicpu_kernels PROPERTIES
    OUTPUT_NAME transformer_aicpu_kernels
)

target_compile_definitions(cust_aicpu_kernels PRIVATE
    _GLIBCXX_USE_CXX11_ABI=1
)

target_include_directories(cust_aicpu_kernels PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include/aicpu
)

target_compile_options(cust_aicpu_kernels PRIVATE -pthread)

target_link_libraries(cust_aicpu_kernels PRIVATE
    -Wl,--whole-archive
    ${AICPU_LIB_DIR}/libaicpu_context.a
    ${AICPU_LIB_DIR}/libbase_ascend_protobuf.a
    -Wl,--no-whole-archive
    pthread
)

target_link_options(cust_aicpu_kernels PRIVATE
    -s
    -Wl,-Bsymbolic
    -Wl,--exclude-libs=libbase_ascend_protobuf.a
)
