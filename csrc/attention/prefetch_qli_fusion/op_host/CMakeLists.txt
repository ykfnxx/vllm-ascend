add_op_to_compiled_list()

add_ops_compile_options(
        OP_NAME PrefetchQliFusion
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        prefetch_qli_fusion_def.cpp
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE prefetch_qli_fusion ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
