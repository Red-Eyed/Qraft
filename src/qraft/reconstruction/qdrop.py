"""Streamed operator reconstruction with QDrop and adaptive weight rounding.

Based on arXiv:2203.05740. This operator-level variant does not discover
residual blocks or reproduce the paper's cached random minibatch schedule.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import cos, pi, sqrt
from typing import assert_never

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from torch import Tensor
from torch.nn import functional as F

from qraft.domain import Encoding, FloatArray
from qraft.reconstruction import Batch, Conv2d, Linear, Problem, Replay, Solution
from qraft.reconstruction.affine import parameters
from qraft.result import Err, FailureKind, Ok, QraftError, Result, failure, validate

_ARRAY = TypeAdapter(
    NDArray[np.generic], config=ConfigDict(arbitrary_types_allowed=True)
)


def array(tensor: Tensor) -> FloatArray:
    """Validate the NumPy bridge before admitting Torch output as native FP32."""
    value = _ARRAY.validate_python(tensor.detach().cpu().numpy())
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("optimizer returned nonfinite or non-FP32 values")
    return np.asarray(value, dtype=np.float32)


def forward(operator: Linear | Conv2d, inputs: Tensor, weight: Tensor) -> Tensor:
    """Apply a bias-free operator; unchanged additive bias cancels in its loss."""
    match operator:
        case Linear():
            return F.linear(inputs, weight)
        case Conv2d(strides=strides, dilations=dilations, pads=pads, groups=groups):
            top, left, bottom, right = pads
            padded = F.pad(inputs, (left, right, top, bottom))
            return F.conv2d(
                padded, weight, stride=strides, dilation=dilations, groups=groups
            )
        case _:
            assert_never(operator)


def soft_round(alpha: Tensor) -> Tensor:
    """Use AdaRound's stretched sigmoid with hard zero/one endpoints."""
    return (torch.sigmoid(alpha) * 1.2 - 0.1).clamp(0, 1)


def straight_round(value: Tensor) -> Tensor:
    """Round in the forward pass and preserve the identity surrogate gradient."""
    return value + (value.round() - value).detach()


def activation_quantize(
    inputs: Tensor, scale: Tensor, zero: int, low: int, high: int
) -> Tensor:
    """Use an LSQ-style normalized scale gradient and fixed integer zero point."""
    factor = 1 / sqrt(inputs.numel() * max(abs(low), abs(high)))
    scaled = scale * factor + (scale - scale * factor).detach()
    quantized = straight_round(inputs / scaled + zero).clamp(low, high)
    return (quantized - zero) * scaled


def mixed_inputs(
    reference: Tensor, candidate: Tensor, probability: float, generator: torch.Generator
) -> Tensor:
    """Keep each quantized activation independently with the specified probability."""
    mask = (
        torch.rand(candidate.shape, generator=generator, device=candidate.device)
        < probability
    )
    return torch.where(mask, candidate, reference)


@dataclass(frozen=True)
class State:
    """Keep owned optimizer tensors separate from immutable domain inputs."""

    weight: Tensor
    scale: Tensor
    zero: Tensor
    floor: Tensor
    alpha: Tensor
    activation_scale: Tensor


def initialize(problem: Problem, device: torch.device) -> State:
    """Initialize soft rounding to exactly recover each unrounded weight."""
    weight = torch.tensor(problem.weight.copy(), device=device)
    scale_array, zero_array = parameters(problem.weight_encoding, weight.ndim)
    scale = torch.tensor(scale_array.copy(), device=device)
    zero = torch.tensor(zero_array.astype(np.float32), device=device)
    normalized = weight / scale
    floor = normalized.floor()
    fraction = (normalized - floor + 0.1) / 1.2
    alpha = torch.log(fraction / (1 - fraction)).detach().requires_grad_()
    activation_scale = torch.tensor(
        problem.activation_encoding.scale.copy(), device=device
    ).requires_grad_()
    return State(weight, scale, zero, floor, alpha, activation_scale)


def check_batch(problem: Problem, batch: Batch) -> Result[Batch, QraftError]:
    """Reject incompatible layouts before entering Torch operator execution."""
    shape = batch.reference.shape
    match problem.operator:
        case Linear():
            valid = len(shape) >= 2 and shape[-1] == problem.weight.shape[1]
        case Conv2d(groups=groups, pads=pads, dilations=dilations):
            valid = len(shape) == 4 and shape[1] == problem.weight.shape[1] * groups
            if valid:
                valid = all(
                    shape[axis + 2] + pads[axis] + pads[axis + 2]
                    >= dilations[axis] * (problem.weight.shape[axis + 2] - 1) + 1
                    for axis in range(2)
                )
        case _:
            assert_never(problem.operator)
    if not valid:
        return failure(
            FailureKind.INVALID_DATA, "QDrop", "input layout incompatible with operator"
        )
    return Ok(batch)


def rounding_penalty(alpha: Tensor, step: int, config: QDrop) -> Tensor:
    """Anneal the rounding regularizer after the configured warmup fraction."""
    progress = (step + 1) / config.steps
    if progress < config.warmup:
        return alpha.new_zeros(())
    fraction = (progress - config.warmup) / (1 - config.warmup)
    beta = config.beta_start + fraction * (config.beta_end - config.beta_start)
    confidence = (2 * soft_round(alpha) - 1).abs()
    return config.rounding_weight * (1 - confidence.pow(beta)).sum()


def objective(
    problem: Problem,
    batch: Batch,
    state: State,
    config: QDrop,
    generator: torch.Generator,
    step: int,
) -> Tensor:
    """Combine reference-output reconstruction with soft-rounding regularization."""
    reference = torch.tensor(batch.reference.copy(), device=state.weight.device)
    candidate = torch.tensor(batch.candidate.copy(), device=state.weight.device)
    activation_bounds = np.iinfo(problem.activation_encoding.zero_point.dtype)
    quantized = activation_quantize(
        candidate,
        state.activation_scale,
        int(problem.activation_encoding.zero_point),
        activation_bounds.min,
        activation_bounds.max,
    )
    inputs = mixed_inputs(reference, quantized, config.keep_probability, generator)
    weight_bounds = np.iinfo(problem.weight_encoding.zero_point.dtype)
    weight_codes = (state.floor + soft_round(state.alpha) + state.zero).clamp(
        weight_bounds.min, weight_bounds.max
    )
    weights = (weight_codes - state.zero) * state.scale
    prediction = forward(problem.operator, inputs, weights)
    with torch.no_grad():
        target = forward(problem.operator, reference, state.weight)
    match problem.operator:
        case Linear():
            channel_axis = -1
        case Conv2d():
            channel_axis = 1
    reconstruction = (prediction - target).square().sum(channel_axis).mean()
    return reconstruction + rounding_penalty(state.alpha, step, config)


def finish(problem: Problem, state: State) -> Solution:
    """Harden learned rounding once and preserve learned activation scales."""
    if not bool(torch.isfinite(state.alpha).all()):
        raise ValueError("optimizer returned nonfinite rounding parameters")
    bounds = np.iinfo(problem.weight_encoding.zero_point.dtype)
    values = (state.floor + (state.alpha >= 0).float() + state.zero).clamp(
        bounds.min, bounds.max
    )
    return Solution(
        codes=array(values).astype(np.int8)
        if problem.weight_encoding.zero_point.dtype == np.int8
        else array(values).astype(np.uint8),
        activation_encoding=Encoding(
            scale=array(state.activation_scale),
            zero_point=problem.activation_encoding.zero_point,
            granularity=problem.activation_encoding.granularity,
        ),
    )


def scheduled_batches(
    replay: Replay, steps: int
) -> Iterable[Result[Batch, QraftError]]:
    """Cycle a replay factory without caching or reading beyond the step budget."""
    remaining = steps
    while remaining:
        seen = False
        for outcome in replay():
            seen = True
            yield outcome
            match outcome:
                case Err():
                    return
                case Ok():
                    pass
            remaining -= 1
            if not remaining:
                return
        if not seen:
            yield failure(
                FailureKind.EMPTY, "QDrop", "calibration replay yielded no batches"
            )
            return


def train_step(
    problem: Problem,
    batch: Batch,
    state: State,
    config: QDrop,
    generator: torch.Generator,
    optimizer: torch.optim.Adam,
    step: int,
) -> Result[None, QraftError]:
    """Update owned optimizer state only after admitting a compatible batch."""
    match check_batch(problem, batch):
        case Err() as error:
            return error
        case Ok():
            pass
    optimizer.zero_grad()
    loss = objective(problem, batch, state, config, generator, step)
    if not bool(torch.isfinite(loss)):
        return failure(
            FailureKind.INVALID_DATA, "QDrop", "reconstruction loss is nonfinite"
        )
    loss.backward()
    optimizer.param_groups[1]["lr"] = (
        config.scale_learning_rate * (1 + cos(pi * step / config.steps)) / 2
    )
    optimizer.step()
    with torch.no_grad():
        state.activation_scale.clamp_(min=torch.finfo(torch.float32).tiny)
    return Ok(None)


class QDrop(BaseModel):
    """Optimize operator rounding and activation scale with streamed QDrop inputs.

    Replay cycles deterministically; the caller controls sample order and size.
    A local random generator leaves application RNG state untouched. CPU is the
    portable default; CUDA may be selected explicitly. Requires qraft[qdrop].
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    steps: int = Field(default=2000, gt=0)
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    scale_learning_rate: float = Field(default=0.00004, gt=0, allow_inf_nan=False)
    keep_probability: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    rounding_weight: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    warmup: float = Field(default=0.2, ge=0, lt=1, allow_inf_nan=False)
    beta_start: float = Field(default=20, ge=1, allow_inf_nan=False)
    beta_end: float = Field(default=2, ge=1, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0)
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda(:[0-9]+)?)$")

    def reconstruct(
        self, problem: Problem, replay: Replay
    ) -> Result[Solution, QraftError]:
        """Optimize owned tensors and return expected data or finite-loss failures."""
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            return failure(FailureKind.UNSUPPORTED, "QDrop", "CUDA is unavailable")
        state = initialize(problem, device)
        generator = torch.Generator(device=device).manual_seed(self.seed)
        optimizer = torch.optim.Adam(
            [
                {"params": [state.alpha], "lr": self.learning_rate},
                {"params": [state.activation_scale], "lr": self.scale_learning_rate},
            ]
        )
        return self._optimize(problem, replay, state, generator, optimizer)

    def _optimize(
        self,
        problem: Problem,
        replay: Replay,
        state: State,
        generator: torch.Generator,
        optimizer: torch.optim.Adam,
    ) -> Result[Solution, QraftError]:
        """Replay without an activation cache, stopping at exactly the step budget."""
        for step, outcome in enumerate(scheduled_batches(replay, self.steps)):
            match outcome:
                case Err() as error:
                    return error
                case Ok(batch):
                    pass
            match train_step(problem, batch, state, self, generator, optimizer, step):
                case Err() as error:
                    return error
                case Ok():
                    pass
        return validate("QDrop solution", lambda: finish(problem, state))
