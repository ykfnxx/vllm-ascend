add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        asu_kv_gather_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME AsuKvGather
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        --op_relocatable_kernel_binary=true
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE asu_kv_gather ACLNNTYPE aclnn)
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE
        asu_kv_gather_proto.cpp
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
