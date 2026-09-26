"""Capture installed runtime versions for example report provenance."""

import platform
from importlib.metadata import version

from examples._shared.schema import Environment


def environment() -> Environment:
    """Capture installed versions rather than inferring them from requirements."""
    return Environment(
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        torch=version("torch"),
        torchvision=version("torchvision"),
        onnx=version("onnx"),
        onnxruntime=version("onnxruntime"),
        qraft=version("qraft"),
        stackformers=version("stackformers"),
    )
