# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

set(_gskvc_shared_dir ${CMAKE_CURRENT_LIST_DIR}/../gather_selection_kv_cache)
include(${_gskvc_shared_dir}/gather_selection_kv_cache_common.cmake)

gskvc_add_shared_host_sources(${_gskvc_shared_dir})

gskvc_enable_op(
        OP_NAME KVSelect
        OP_DEF op_host/kv_select_def.cpp
        BUILD_DEFINE GSKVC_BUILD_KV_SELECT
        SOURCE_ALIAS kv_select
        SOURCE_DIR ${_gskvc_shared_dir}
)
