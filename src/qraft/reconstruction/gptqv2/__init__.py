"""GPTQv2 asymmetric calibration; README.md links Algorithm 1 and its foundations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from qraft.domain import Encoding, FloatArray, IntArray, PerChannel
from qraft.reconstruction import Linear, Problem, Replay, Solution
from qraft.reconstruction.affine import codes, quantize, restore
from qraft.result import Err, FailureKind, Ok, QraftError, Result, failure


@dataclass(frozen=True)
class Moments:
    """Hold input Gram and reference-minus-candidate cross moments."""

    gram: NDArray[np.float64]
    residual: NDArray[np.float64]


def collect(problem: Problem, replay: Replay) -> Result[Moments, QraftError]:
    """Accumulate quadratic statistics without retaining calibration activations."""
    width = problem.weight.shape[1]
    gram = np.zeros((width, width), dtype=np.float64)
    residual = np.zeros_like(gram)
    rows = 0
    for outcome in replay():
        match outcome:
            case Err() as error:
                return error
            case Ok(batch):
                pass
        if batch.reference.ndim < 2 or batch.reference.shape[-1] != width:
            return failure(FailureKind.INVALID_DATA, "GPTQv2", "input channel mismatch")
        reference = batch.reference.reshape(-1, width).astype(np.float64)
        candidate = quantize(batch.candidate, problem.activation_encoding)
        inputs = candidate.reshape(-1, width).astype(np.float64)
        gram += inputs.T @ inputs
        residual += (reference - inputs).T @ inputs
        rows += inputs.shape[0]
    if rows == 0:
        return failure(FailureKind.EMPTY, "GPTQv2", "calibration yielded no batches")
    gram /= rows
    residual /= rows
    if not np.isfinite(gram).all() or not np.isfinite(residual).all():
        return failure(
            FailureKind.INVALID_DATA, "GPTQv2", "calibration moments overflowed"
        )
    return Ok(Moments(gram, residual))


def reconstruct_weights(
    weight: FloatArray, encoding: Encoding, moments: Moments, damping: float
) -> IntArray:
    """Apply columnwise rounding compensation and asymmetric residual correction.

    This is the unblocked form of Algorithm 1. The lower Cholesky factor's
    trailing submatrices represent successive Schur complements; transposing
    the factor before computing the residual correction would change the method.
    """
    gram = moments.gram.copy()
    diagonal = np.diag_indices_from(gram)
    gram[diagonal] += damping * max(float(np.mean(np.diag(gram))), 1e-12)
    lower = np.linalg.cholesky(np.linalg.inv(gram))
    correction = np.triu(moments.residual @ lower, k=1) @ lower.T
    working = weight.astype(np.float64)
    result = np.empty(weight.shape, dtype=encoding.zero_point.dtype)
    column_encoding = Encoding(
        scale=encoding.scale,
        zero_point=encoding.zero_point,
        granularity=PerChannel(axis=0),
    )
    for column in range(weight.shape[1]):
        current = working[:, column].copy()
        integer = codes(current, column_encoding)
        rounded = restore(integer, column_encoding).astype(np.float64)
        result[:, column] = integer
        error = (current - rounded) / lower[column, column]
        working[:, column:] -= np.outer(error, lower[column:, column])
        working[:, column:] += np.outer(current, correction[column, column:])
    if not np.isfinite(working).all():
        raise ValueError("GPTQv2 weight updates overflowed")
    return result


class GPTQv2(BaseModel):
    """Reconstruct linear weights with streamed asymmetric input statistics.

    Memory grows quadratically with input width, independently of sample count.
    The workspace limit estimates twelve float64 square matrices and rejects
    oversized layers before allocating their statistics. It excludes model and
    current-batch storage. Export remains standard affine INT8/UINT8.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    damping: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    max_workspace_bytes: int = Field(default=1_073_741_824, gt=0)

    def reconstruct(
        self, problem: Problem, replay: Replay
    ) -> Result[Solution, QraftError]:
        """Return integer weights or expected layout, replay, or numerical failures."""
        match problem.operator:
            case Linear():
                pass
            case _:
                return failure(
                    FailureKind.UNSUPPORTED, "GPTQv2", "linear operators required"
                )
        if 12 * 8 * problem.weight.shape[1] ** 2 > self.max_workspace_bytes:
            return failure(
                FailureKind.UNSUPPORTED,
                "GPTQv2",
                "quadratic workspace exceeds configured limit",
            )
        match collect(problem, replay):
            case Err() as error:
                return error
            case Ok(moments):
                pass
        try:
            values = reconstruct_weights(
                problem.weight, problem.weight_encoding, moments, self.damping
            )
        except (np.linalg.LinAlgError, ValueError) as error:
            return failure(FailureKind.INVALID_DATA, "GPTQv2", str(error))
        return Ok(
            Solution(codes=values, activation_encoding=problem.activation_encoding)
        )
