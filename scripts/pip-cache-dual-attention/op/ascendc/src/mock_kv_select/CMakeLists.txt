# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

set(_mock_kv_select_dir ${CMAKE_CURRENT_LIST_DIR}/../../../aicpu_mock_kv_select)

target_sources(op_host_aclnn PRIVATE
        ${_mock_kv_select_dir}/op_host/mock_kv_select_def.cpp
)

target_sources(opsproto PRIVATE
        ${_mock_kv_select_dir}/op_proto/mock_kv_select_proto.cpp
)
