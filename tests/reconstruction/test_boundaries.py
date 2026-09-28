"""Admission, ownership, and replay failure contracts for reconstruction."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from qraft.domain import Encoding, FloatArray, InputArray, PerChannel
from qraft.plan import QuantizationPlan, QuantizeConstant, QuantizeInput
from qraft.reconstruction import Batch, Problem, Replay
from qraft.reconstruction.gptqv2 import GPTQv2
from qraft.reconstruction.qdrop import QDrop
from qraft.reconstruction.replay import paired_replay
from qraft.result import Err, FailureKind, Ok, QraftError, Result, failure


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[np.nan]], dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float64),
    ],
)
def test_batch_rejects_invalid_arrays(invalid: NDArray[np.generic]) -> None:
    """Reject nonfinite, empty, and wrong-dtype external responses at admission."""
    with pytest.raises(ValidationError):
        Batch.model_validate({"reference": invalid, "candidate": invalid})


def test_batch_owns_storage() -> None:
    """Freezing the admitted record must not freeze or retain caller arrays."""
    source = np.ones((2, 3), dtype=np.float32)
    batch = Batch(reference=source, candidate=source)
    source[:] = 7
    assert source.flags.writeable
    assert not batch.reference.flags.writeable
    assert not batch.candidate.flags.writeable
    np.testing.assert_array_equal(batch.reference, 1)


def test_constant_ownership_and_conflicts() -> None:
    """An exact-code operation owns its payload and conflicts with QDQ on that edge."""
    source = np.ones((2, 3), dtype=np.int8)
    encoding = Encoding(
        scale=np.ones(2, dtype=np.float32),
        zero_point=np.zeros(2, dtype=np.int8),
        granularity=PerChannel(axis=0),
    )
    constant = QuantizeConstant(node="n", index=1, values=source, encoding=encoding)
    source[:] = 9
    np.testing.assert_array_equal(constant.values, 1)
    assert not constant.values.flags.writeable
    with pytest.raises(ValidationError, match="duplicate"):
        QuantizationPlan(
            operations=(constant, QuantizeInput(node="n", index=1, encoding=encoding))
        )


@dataclass(frozen=True)
class MissingTensor:
    """Simulate an evaluator that violates its requested-output contract."""

    def run(
        self, sample: Mapping[str, InputArray], outputs: tuple[str, ...]
    ) -> Result[Mapping[str, FloatArray], QraftError]:
        """Return no tensors so replay admission must reject the response."""
        return Ok({})


def test_missing_runtime_output_is_an_error() -> None:
    """Missing keys become explicit admission failures rather than KeyError."""
    replay = paired_replay(
        MissingTensor(), MissingTensor(), lambda: ({},), "x", lambda value: value
    )
    outcome = next(iter(replay()))
    assert isinstance(outcome, Err)
    assert outcome.error.kind == FailureKind.INVALID_DATA


@pytest.mark.parametrize("method", [GPTQv2(), QDrop(steps=1)])
def test_unexpected_plugin_exceptions_propagate(
    problem: Problem, method: GPTQv2 | QDrop
) -> None:
    """A programming failure must not be relabeled as invalid calibration data."""

    def broken() -> Replay:
        """Return a source whose execution raises an unexpected exception."""

        def source() -> tuple[Result[Batch, QraftError], ...]:
            """Represent a plugin defect, not a documented evaluator failure."""
            raise RuntimeError("plugin defect")

        return source

    with pytest.raises(RuntimeError, match="plugin defect"):
        method.reconstruct(problem, broken())


@pytest.mark.parametrize("method", [GPTQv2(), QDrop(steps=1)])
def test_wrong_channel_layout(problem: Problem, method: GPTQv2 | QDrop) -> None:
    """Reject bad channels before numerical library matrix operations."""
    batch = Batch(
        reference=np.ones((2, 4), dtype=np.float32),
        candidate=np.ones((2, 4), dtype=np.float32),
    )
    result = method.reconstruct(problem, lambda: (Ok(batch),))
    assert isinstance(result, Err)
    assert result.error.kind == FailureKind.INVALID_DATA


def test_replay_failure_identity() -> None:
    """Forward expected execution failures unchanged from either graph revision."""
    error = failure(FailureKind.EXECUTION, "runtime", "failed")

    @dataclass(frozen=True)
    class FailingEvaluator:
        """Provide an injected runtime failure without involving ONNX."""

        def run(
            self, sample: Mapping[str, InputArray], outputs: tuple[str, ...]
        ) -> Result[Mapping[str, FloatArray], QraftError]:
            """Return the shared failure container."""
            return error

    replay = paired_replay(
        FailingEvaluator(), MissingTensor(), lambda: ({},), "x", lambda value: value
    )
    assert next(iter(replay())) is error
