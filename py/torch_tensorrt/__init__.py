import ctypes
import os
import platform
import sys
from typing import Dict, List

from torch_tensorrt._version import (  # noqa: F401
    __cuda_version__,
    __cudnn_version__,
    __tensorrt_version__,
    __version__,
)

from packaging import version

if sys.version_info < (3,):
    raise Exception(
        "Python 2 has reached end-of-life and is not supported by Torch-TensorRT"
    )


def _parse_semver(version: str) -> Dict[str, str]:
    split = version.split(".")
    if len(split) < 3:
        split.append("")

    return {"major": split[0], "minor": split[1], "patch": split[2]}


def _find_lib(name: str, paths: List[str]) -> str:
    for path in paths:
        libpath = os.path.join(path, name)
        if os.path.isfile(libpath):
            return libpath

    raise FileNotFoundError(f"Could not find {name}\n  Search paths: {paths}")


try:
    import tensorrt  # noqa: F401
except ImportError:
    cuda_version = _parse_semver(__cuda_version__)
    cudnn_version = _parse_semver(__cudnn_version__)
    tensorrt_version = _parse_semver(__tensorrt_version__)

    CUDA_MAJOR = cuda_version["major"]
    CUDNN_MAJOR = cudnn_version["major"]
    TENSORRT_MAJOR = tensorrt_version["major"]

    if sys.platform.startswith("win"):
        WIN_LIBS = [
            "nvinfer.dll",
            "nvinfer_plugin.dll",
        ]

        WIN_PATHS = os.environ["PATH"].split(os.path.pathsep)

        for lib in WIN_LIBS:
            ctypes.CDLL(_find_lib(lib, WIN_PATHS))

    elif sys.platform.startswith("linux"):
        LINUX_PATHS = ["/usr/local/cuda-12.4/lib64", "/usr/lib", "/usr/lib64"]
        if "LD_LIBRARY_PATH" in os.environ:
            LINUX_PATHS += os.environ["LD_LIBRARY_PATH"].split(os.path.pathsep)

        if platform.uname().processor == "x86_64":
            LINUX_PATHS += [
                "/usr/lib/x86_64-linux-gnu",
            ]

        elif platform.uname().processor == "aarch64":
            LINUX_PATHS += ["/usr/lib/aarch64-linux-gnu"]

        LINUX_LIBS = [
            f"libnvinfer.so.{TENSORRT_MAJOR}",
            f"libnvinfer_plugin.so.{TENSORRT_MAJOR}",
        ]

        for lib in LINUX_LIBS:
            ctypes.CDLL(_find_lib(lib, LINUX_PATHS))

import logging

import torch
from torch_tensorrt._features import ENABLED_FEATURES, _enabled_features_str

_LOGGER = logging.getLogger(__name__)
_LOGGER.debug(_enabled_features_str())


def _register_with_torch() -> None:
    trtorch_dir = os.path.dirname(__file__)
    linked_file = (
        "/torchtrt.dll" if sys.platform.startswith("win") else "/lib/libtorchtrt.so"
    )
    linked_file_runtime = (
        "/torchtrt_runtime.dll"
        if sys.platform.startswith("win")
        else "/lib/libtorchtrt_runtime.so"
    )
    if os.path.isfile(trtorch_dir + linked_file):
        assert ENABLED_FEATURES.torchscript_frontend
        assert ENABLED_FEATURES.torch_tensorrt_runtime
        torch.ops.load_library(trtorch_dir + linked_file)
    elif os.path.isfile(trtorch_dir + linked_file_runtime):
        assert ENABLED_FEATURES.torch_tensorrt_runtime
        torch.ops.load_library(trtorch_dir + linked_file_runtime)


_register_with_torch()

from torch_tensorrt._Device import Device  # noqa: F401
from torch_tensorrt._enums import (  # noqa: F401
    DeviceType,
    EngineCapability,
    dtype,
    memory_format,
)
from torch_tensorrt._Input import Input  # noqa: F401
from torch_tensorrt.runtime import *  # noqa: F403

if ENABLED_FEATURES.torchscript_frontend:
    from torch_tensorrt import ts

if ENABLED_FEATURES.fx_frontend:
    from torch_tensorrt import fx

if ENABLED_FEATURES.dynamo_frontend:
    from torch_tensorrt.dynamo import backend  # noqa: F401
    from torch_tensorrt import dynamo  # noqa: F401

from torch_tensorrt._compile import *  # noqa: F403
