"""Independent numerical and streaming checks for asymmetric GPTQv2."""

import weakref
from collections.abc import Iterable

import numpy as np
import pytest

from quantsmith.domain import IntArray
from quantsmith.reconstruction import Batch, Problem, Replay
from quantsmith.reconstruction.affine import quantize, restore
from quantsmith.reconstruction.gptqv2 import GPTQv2, collect
from quantsmith.result import FailureKind, Ok, QuantSmithError, Result, failure
from tests.outcomes import expect_ok


def direct_solution(problem: Problem, batch: Batch, damping: float) -> IntArray:
    """Solve each trailing constrained problem directly, without Cholesky fusion."""
    inputs = quantize(batch.candidate, problem.activation_encoding).astype(np.float64)
    reference = batch.reference.astype(np.float64)
    gram = inputs.T @ inputs / len(inputs)
    residual = (reference - inputs).T @ inputs / len(inputs)
    gram += np.eye(gram.shape[0]) * damping * np.diag(gram).mean()
    weight = problem.weight.astype(np.float64)
    result = np.zeros(weight.shape, dtype=np.int8)
    scale = problem.weight_encoding.scale
    for column in range(weight.shape[1]):
        current = weight[:, column].copy()
        integer = np.clip(np.rint(current / scale), -128, 127).astype(np.int8)
        result[:, column] = integer
        inverse = np.linalg.inv(gram[column:, column:])
        weight[:, column:] += np.outer(
            integer * scale - current, inverse[0] / inverse[0, 0]
        )
        tail = column + 1
        if tail < weight.shape[1]:
            correction = residual[column, tail:] @ np.linalg.inv(gram[tail:, tail:])
            weight[:, tail:] += np.outer(current, correction)
    return result


def test_matches_independent_constrained_solution(
    problem: Problem, batch: Batch, replay: Replay
) -> None:
    """Catch Cholesky orientation, residual sign, and column-update mistakes."""
    actual = expect_ok(GPTQv2().reconstruct(problem, replay))
    np.testing.assert_array_equal(actual.codes, direct_solution(problem, batch, 0.01))


def test_asymmetric_correction_improves_reference_error(
    problem: Problem, asymmetric_batch: Batch
) -> None:
    """Compare the same grid with and without upstream-error compensation."""
    batch = asymmetric_batch
    actual = expect_ok(GPTQv2().reconstruct(problem, lambda: (Ok(batch),)))
    np.testing.assert_array_equal(actual.codes, direct_solution(problem, batch, 0.01))
    inputs = quantize(batch.candidate, problem.activation_encoding)
    target = batch.reference @ problem.weight.T
    corrected = inputs @ restore(actual.codes, problem.weight_encoding).T
    nearest = inputs @ quantize(problem.weight, problem.weight_encoding).T
    assert np.mean((corrected - target) ** 2) < np.mean((nearest - target) ** 2)


def test_statistics_are_partition_invariant(
    problem: Problem, batch: Batch, replay: Replay
) -> None:
    """Weight moments by row count, including differently sized final batches."""

    def chunks() -> Iterable[Result[Batch, QuantSmithError]]:
        """Partition the same samples without changing their contributions."""
        for start in range(0, len(batch.reference), 7):
            yield Ok(
                Batch(
                    reference=batch.reference[start : start + 7],
                    candidate=batch.candidate[start : start + 7],
                )
            )

    full = expect_ok(collect(problem, replay))
    split = expect_ok(collect(problem, chunks))
    np.testing.assert_allclose(full.gram, split.gram, atol=1e-14)
    np.testing.assert_allclose(full.residual, split.residual, atol=1e-14)


def test_stream_does_not_retain_prior_batches(problem: Problem) -> None:
    """Track live arrays while increasing sample count at fixed batch size."""
    refs: list[weakref.ReferenceType[Batch]] = []

    def stream() -> Iterable[Result[Batch, QuantSmithError]]:
        """Permit at most the previous and current batches to remain live."""
        for _ in range(40):
            batch = Batch(
                reference=np.ones((2, 3), dtype=np.float32),
                candidate=np.ones((2, 3), dtype=np.float32),
            )
            refs.append(weakref.ref(batch))
            assert sum(ref() is not None for ref in refs) <= 2
            yield Ok(batch)

    expect_ok(GPTQv2().reconstruct(problem, stream))
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("method", [GPTQv2(), GPTQv2(max_workspace_bytes=1)])
def test_empty_and_workspace_failures(problem: Problem, method: GPTQv2) -> None:
    """Reject empty data or insufficient workspace as expected outcomes."""
    from quantsmith.result import Err

    result = method.reconstruct(problem, lambda: ())
    assert isinstance(result, Err)
    assert result.error.kind in (FailureKind.EMPTY, FailureKind.UNSUPPORTED)


def test_replay_failure_preserves_diagnostic(problem: Problem) -> None:
    """Do not wrap an evaluator's error in a generic reconstruction diagnostic."""
    error = failure(FailureKind.EXECUTION, "fake evaluator", "unavailable")
    assert GPTQv2().reconstruct(problem, lambda: (error,)) is error
