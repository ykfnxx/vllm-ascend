# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

function(gskvc_add_compile_options op_name)
    add_ops_compile_options(
            OP_NAME ${op_name}
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
                    -mllvm -cce-aicore-hoist-movemask=false
                    --op_relocatable_kernel_binary=true
    )
endfunction()

function(gskvc_add_shared_host_sources shared_src_dir)
    get_filename_component(_shared_src_dir "${shared_src_dir}" ABSOLUTE)
    get_property(_shared_sources_added GLOBAL PROPERTY GSKVC_SHARED_HOST_SOURCES_ADDED)
    if (_shared_sources_added)
        return()
    endif()

    target_sources(optiling PRIVATE
            ${_shared_src_dir}/op_host/gather_selection_kv_cache_tiling.cpp
    )

    if (NOT BUILD_OPEN_PROJECT)
        target_sources(opmaster_ct PRIVATE
                ${_shared_src_dir}/op_host/gather_selection_kv_cache_tiling.cpp
        )
    endif ()

    target_include_directories(optiling PRIVATE
            ${_shared_src_dir}/op_host
    )

    target_sources(opsproto PRIVATE
            ${_shared_src_dir}/op_host/gather_selection_kv_cache_proto.cpp
    )

    set_property(GLOBAL PROPERTY GSKVC_SHARED_HOST_SOURCES_ADDED TRUE)
endfunction()

function(gskvc_enable_op)
    cmake_parse_arguments(GSKVC "" "OP_NAME;OP_DEF;BUILD_DEFINE;SOURCE_ALIAS;SOURCE_DIR" "" ${ARGN})

    get_filename_component(_source_dir "${GSKVC_SOURCE_DIR}" ABSOLUTE)
    gskvc_add_compile_options(${GSKVC_OP_NAME})

    target_sources(op_host_aclnn PRIVATE
            ${_source_dir}/${GSKVC_OP_DEF}
    )

    target_compile_definitions(optiling PRIVATE ${GSKVC_BUILD_DEFINE}=1)
    target_compile_definitions(opsproto PRIVATE ${GSKVC_BUILD_DEFINE}=1)
    if (TARGET opmaster_ct)
        target_compile_definitions(opmaster_ct PRIVATE ${GSKVC_BUILD_DEFINE}=1)
    endif()

    if (DEFINED GSKVC_SOURCE_ALIAS AND NOT "${GSKVC_SOURCE_ALIAS}" STREQUAL "")
        set(${GSKVC_SOURCE_ALIAS}_dir ${_source_dir}
            CACHE INTERNAL "selection kv cache shared kernel source dir" FORCE)
        set(${GSKVC_SOURCE_ALIAS}_source_dir ${_source_dir}
            CACHE INTERNAL "selection kv cache binary compile source dir" FORCE)
    endif()
endfunction()
