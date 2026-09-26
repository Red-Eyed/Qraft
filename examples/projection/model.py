"""Deterministic FP32 projection without downloaded data or weights."""

from collections.abc import Iterable, Mapping

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

from qraft.calibration import Samples
from qraft.domain import FloatArray


def example_model() -> ModelProto:
    """Construct a small floating-point projection without filesystem inputs."""
    weight = np.random.default_rng(7).normal(size=(8, 4)).astype(np.float32)
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"], name="projection")],
            "projection",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])],
            [numpy_helper.from_array(weight, "w")],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )


def sample_source(seed: int, batches: int) -> Samples:
    """Bind replay parameters so each pass regenerates the same streamed batches."""

    def samples() -> Iterable[Mapping[str, FloatArray]]:
        """Retain only the current sixteen-row batch of eight features."""
        rng = np.random.default_rng(seed)
        for _ in range(batches):
            yield {"x": rng.normal(size=(16, 8)).astype(np.float32)}

    return samples
