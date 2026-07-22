import glob
import multiprocessing
import os

import torch_npu
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, verify_ninja_availability
from torch_npu.utils.cpp_extension import NpuExtension


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))

USE_NINJA = os.getenv("USE_NINJA") == "1"
MAX_JOBS = int(os.getenv("MAX_JOBS", multiprocessing.cpu_count()))

if USE_NINJA:
    verify_ninja_availability()

source_files = glob.glob(os.path.join(BASE_DIR, "csrc", "*.cpp"))

setup(
    name="lightning_indexer_decode_custom_ops",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        NpuExtension(
            name="lightning_indexer_decode_custom_ops.custom_ops_lib",
            sources=source_files,
            extra_compile_args=[
                "-I" + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
                "-O3",
                "-std=c++17",
                "-fvisibility=hidden",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA, parallel=MAX_JOBS)},
)
