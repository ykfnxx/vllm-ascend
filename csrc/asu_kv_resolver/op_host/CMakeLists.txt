add_ops_compile_options(
        OP_NAME AsuKvResolver
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        asu_kv_resolver_def.cpp
)

target_sources(optiling PRIVATE
        asu_kv_resolver_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        asu_kv_resolver_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        asu_kv_resolver_infershape.cpp
)
