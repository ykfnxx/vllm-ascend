#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_connector():
    from vllm_ascend.distributed.kv_transfer import register_connector

    register_connector()


def register_dsa_sparse():
    """Install DSA runtime patches in each vLLM process."""
    from vllm.logger import init_logger
    from vllm_ascend.patch.dsa_sparse.patch_runtime import (
        install_dsa_runtime_patches,
    )

    install_dsa_runtime_patches()
    init_logger("vllm.dsa_sparse").info_once(
        "DSA sparse general-plugin bootstrap installed"
    )


def register_model_loader():
    from .model_loader.netloader import register_netloader
    from .model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    from .profiling_config import generate_service_profiling_config

    generate_service_profiling_config()
