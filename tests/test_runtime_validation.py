"""Reject malformed runtime responses before exposing typed numerical outputs."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from onnx import ModelProto, TensorProto, helper

from examples._shared.vision.models import OrtRunner
from qraft.domain import FloatArray
from qraft.result import FailureKind
from qraft.runtime import OnnxEvaluator
from tests.outcomes import expect_error, expect_ok


@pytest.fixture
def identity_model() -> ModelProto:
    """Provide real tensor metadata and a session that requires no model download."""
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["images"], ["logits"], name="identity")],
            "identity",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 1000])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1000])],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )


@pytest.fixture
def runtime_runners(
    identity_model: ModelProto, tmp_path: Path
) -> tuple[OnnxEvaluator, OrtRunner]:
    """Build core and example adapters over the same real ONNX graph."""
    path = tmp_path / "identity.onnx"
    path.write_bytes(identity_model.SerializeToString())
    return OnnxEvaluator(identity_model), OrtRunner(path, threads=1)


@pytest.mark.parametrize(
    "payload",
    [
        "not a sequence",
        (np.ones((1, 1000), dtype=np.float32),),
        list[FloatArray](),
        [np.ones((1, 1000), dtype=np.float32)] * 2,
        ["not an array"],
        [np.ones((1, 1000), dtype=np.float64)],
        [np.empty((0,), dtype=np.float32)],
        [np.full((1, 1000), np.nan, dtype=np.float32)],
    ],
)
def test_runtime_response_rejection(
    runtime_runners: tuple[OnnxEvaluator, OrtRunner],
    payload: str | tuple[FloatArray, ...] | list[NDArray[np.generic] | str],
) -> None:
    """Validate containers, array instances, output counts, dtype, and finiteness."""
    evaluator, runner = runtime_runners
    inputs = np.zeros((1, 1000), dtype=np.float32)
    with patch("onnxruntime.InferenceSession.run", return_value=payload):
        error = expect_error(evaluator.run({"images": inputs}, ("logits",)), "")
        assert error.kind == FailureKind.INVALID_DATA
        assert error.detail
        with pytest.raises(ValueError):
            runner(inputs)


def test_runtime_preserves_valid_arrays(
    runtime_runners: tuple[OnnxEvaluator, OrtRunner],
) -> None:
    """Boundary schemas preserve successful execution and numerical outputs."""
    evaluator, runner = runtime_runners
    inputs = np.linspace(-1, 1, 1000, dtype=np.float32)[None, :]
    outputs = expect_ok(evaluator.run({"images": inputs}, ("logits",)))
    np.testing.assert_array_equal(outputs["logits"], inputs)
    np.testing.assert_array_equal(runner(inputs), inputs)
