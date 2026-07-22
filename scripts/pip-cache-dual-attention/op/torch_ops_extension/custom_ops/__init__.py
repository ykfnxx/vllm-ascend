# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import pkgutil
import warnings

__all__ = list(module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)]))

"""
import custom ops as torch_npu ops to support the following usage:
'torch.ops.custom.npu_gather_selection_kv_cache()'
'torch_npu.npu_gather_selection_kv_cache()'
"""

# custom_ops_lib registers PrivateUse1 ops; load torch_npu before the .so.
def _split_env_paths(value):
    return [path for path in value.split(os.pathsep) if path]


def _candidate_opp_paths():
    paths = []
    ascend_opp_path = os.getenv("ASCEND_OPP_PATH")
    if ascend_opp_path:
        paths.append(ascend_opp_path)

    ascend_home_path = os.getenv("ASCEND_HOME_PATH")
    if ascend_home_path:
        paths.append(os.path.join(ascend_home_path, "opp"))

    paths.extend((
        "/usr/local/Ascend/cann-8.5.1/opp",
        "/usr/local/Ascend/ascend-toolkit/latest/opp",
    ))

    seen = set()
    for path in paths:
        norm_path = os.path.normpath(path)
        if norm_path in seen:
            continue
        seen.add(norm_path)
        yield norm_path


def _load_priority_vendors(opp_path):
    config_path = os.path.join(opp_path, "vendors", "config.ini")
    vendors = []
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            for line in config_file:
                line = line.strip()
                if line.startswith("load_priority="):
                    vendors = [vendor.strip() for vendor in line.split("=", 1)[1].split(",") if vendor.strip()]
                    break
    except OSError:
        pass

    preferred_vendors = ("customize_asn",)
    vendors = list(preferred_vendors) + [vendor for vendor in vendors if vendor not in preferred_vendors]
    return vendors


def _has_mock_kv_select_vendor(vendor_root):
    opapi_path = os.path.join(vendor_root, "op_api", "lib", "libcust_opapi.so")
    header_path = os.path.join(vendor_root, "op_api", "include", "aclnn_mock_kv_select.h")
    config_path = os.path.join(vendor_root, "op_impl", "cpu", "config", "cust_aicpu_kernel.json")
    if not (os.path.isfile(opapi_path) and os.path.isfile(header_path) and os.path.isfile(config_path)):
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            return "MockKVSelect" in config_file.read()
    except OSError:
        return False


def _ensure_mock_kv_select_opp_path():
    custom_opp_path = os.getenv("ASCEND_CUSTOM_OPP_PATH", "")
    custom_opp_paths = _split_env_paths(custom_opp_path)
    if any(_has_mock_kv_select_vendor(path) for path in custom_opp_paths):
        return

    for opp_path in _candidate_opp_paths():
        for vendor_name in _load_priority_vendors(opp_path):
            vendor_root = os.path.join(opp_path, "vendors", vendor_name)
            if not _has_mock_kv_select_vendor(vendor_root):
                continue
            paths = [vendor_root] + custom_opp_paths
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(paths)
            return


_ensure_mock_kv_select_opp_path()

import torch
import torch_npu

from . import custom_ops_lib

custom_ops_module = getattr(torch.ops, 'custom', None)
if custom_ops_module is not None:
    if hasattr(custom_ops_module, 'npu_dmp_sparse_flash_attention'):
        from .converter import npu_dmp_sparse_flash_attention  # noqa: F401
    if hasattr(custom_ops_module, 'npu_gather_selection_kv_cache'):
        from .converter import npu_gather_selection_kv_cache  # noqa: F401
    if hasattr(custom_ops_module, 'npu_da_attention_merge'):
        from .converter import npu_da_attention_merge  # noqa: F401
elif os.getenv("CUSTOM_OPS_SFA_ONLY") == "1":
    from .converter import npu_dmp_sparse_flash_attention, npu_da_attention_merge  # noqa: F401
elif os.getenv("CUSTOM_OPS_GATHER_ONLY") == "1":
    from .converter import npu_gather_selection_kv_cache  # noqa: F401
else:
    from .converter import npu_gather_selection_kv_cache, npu_dmp_sparse_flash_attention, npu_da_attention_merge  # noqa: F401

if custom_ops_module is not None:
    known_ops = (
        'npu_gather_selection_kv_cache',
        'npu_gather_selection_kv_cache_functional',
        'npu_kv_select_out',
        'npu_kv_gather_out',
        'npu_mock_kv_select_out',
        'npu_dmp_sparse_flash_attention',
        'npu_da_attention_merge',
    )
    for op_name in known_ops:
        if hasattr(custom_ops_module, op_name):
            setattr(torch_npu, op_name, getattr(custom_ops_module, op_name))

    for op_name in dir(custom_ops_module):
        if op_name.startswith('_') or hasattr(torch_npu, op_name):
            continue
        setattr(torch_npu, op_name, getattr(custom_ops_module, op_name))

else:
    warn_msg = "torch.ops.custom module is not found, mount custom ops to torch_npu failed." \
               "Calling by torch_npu.xxx for custom ops is unsupported, please use torch.ops.custom.xxx."
    warnings.warn(warn_msg)
    warnings.filterwarnings("ignore", message=warn_msg)
