"""QDrop stochastic boundaries, hard reconstruction, and bounded replay."""

from collections.abc import Iterable

import numpy as np
import pytest
import torch

from qraft.reconstruction import Batch, Problem, Replay
from qraft.reconstruction.affine import quantize, restore
from qraft.reconstruction.qdrop import QDrop, mixed_inputs
from qraft.result import Err, FailureKind, Ok, QraftError, Result, failure
from tests.outcomes import expect_ok


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_dropout_endpoints(probability: float) -> None:
    """Dropping everything restores reference inputs; keeping everything uses QDQ."""
    original = torch.ones(4, 3)
    candidate = torch.zeros(4, 3)
    actual = mixed_inputs(
        original, candidate, probability, torch.Generator().manual_seed(3)
    )
    torch.testing.assert_close(actual, candidate if probability == 1 else original)


def test_reproducible_without_global_rng_mutation(
    problem: Problem, replay: Replay
) -> None:
    """Local seeds control dropout without perturbing application randomness."""
    before = torch.random.get_rng_state().clone()
    method = QDrop(steps=10)
    first = expect_ok(method.reconstruct(problem, replay))
    second = expect_ok(method.reconstruct(problem, replay))
    np.testing.assert_array_equal(first.codes, second.codes)
    np.testing.assert_array_equal(
        first.activation_encoding.scale, second.activation_encoding.scale
    )
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    assert not first.codes.flags.writeable
    assert not problem.weight.flags.writeable


def test_hard_reconstruction_improves_error(
    problem: Problem, batch: Batch, replay: Replay
) -> None:
    """Measure exported hard weights, not the optimizer's relaxed training loss."""
    method = QDrop(
        steps=400,
        learning_rate=0.03,
        rounding_weight=0.00001,
        scale_learning_rate=0.00001,
    )
    solution = expect_ok(method.reconstruct(problem, replay))
    target = batch.reference @ problem.weight.T
    nearest = (
        quantize(batch.candidate, problem.activation_encoding)
        @ quantize(problem.weight, problem.weight_encoding).T
    )
    prediction = (
        quantize(batch.candidate, solution.activation_encoding)
        @ restore(solution.codes, problem.weight_encoding).T
    )
    assert np.mean((prediction - target) ** 2) < np.mean((nearest - target) ** 2)


def test_replay_stops_exactly_at_budget(problem: Problem, batch: Batch) -> None:
    """An unbounded source must neither be materialized nor read past the budget."""
    consumed = 0

    def replay() -> Iterable[Result[Batch, QraftError]]:
        """Fail immediately if the optimizer requests an unnecessary batch."""
        nonlocal consumed
        for _ in range(3):
            consumed += 1
            yield Ok(batch)
        raise AssertionError("read past explicit step budget")

    expect_ok(QDrop(steps=3).reconstruct(problem, replay))
    assert consumed == 3


def test_empty_replay_and_execution_failure(problem: Problem) -> None:
    """Empty replays terminate; evaluator errors retain their original identity."""
    result = QDrop(steps=1).reconstruct(problem, lambda: ())
    assert isinstance(result, Err)
    assert result.error.kind == FailureKind.EMPTY
    error = failure(FailureKind.EXECUTION, "fake", "unavailable")
    assert QDrop(steps=1).reconstruct(problem, lambda: (error,)) is error
