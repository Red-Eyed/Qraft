"""Reconstruction contracts shared by independent numerical methods."""

from collections.abc import Callable, Iterable
from typing import Protocol, Self, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quantsmith.domain import (
    Encoding,
    FloatArray,
    IntArray,
    PerChannel,
    PerTensor,
    frozen_array,
)
from quantsmith.result import QuantSmithError, Result


class Batch(BaseModel):
    """Own aligned reference and candidate inputs for one reconstruction unit."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    reference: FloatArray = Field()
    candidate: FloatArray = Field()

    @field_validator("reference", "candidate")
    @classmethod
    def own_array(cls, value: FloatArray) -> FloatArray:
        """Admit finite FP32 values without retaining caller storage."""
        return frozen_array(value)

    @model_validator(mode="after")
    def aligned(self) -> Self:
        """Reject mismatched batch layouts before numerical execution."""
        if self.reference.shape != self.candidate.shape:
            raise ValueError("reference and candidate input shapes must match")
        return self


Replay = Callable[[], Iterable[Result[Batch, QuantSmithError]]]


class Linear(BaseModel):
    """Apply a matrix stored in output-by-input channel order."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class Conv2d(BaseModel):
    """Describe explicit NCHW convolution geometry independently of ONNX."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    strides: tuple[int, int] = (1, 1)
    dilations: tuple[int, int] = (1, 1)
    pads: tuple[int, int, int, int] = (0, 0, 0, 0)
    groups: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def valid_geometry(self) -> Self:
        """Require positive strides/dilations and nonnegative explicit padding."""
        if min(*self.strides, *self.dilations) <= 0 or min(self.pads) < 0:
            raise ValueError("invalid convolution geometry")
        return self


class Problem(BaseModel):
    """Own one operator's weights and affine grids, with output channels first."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    weight: FloatArray = Field()
    weight_encoding: Encoding = Field()
    activation_encoding: Encoding = Field()
    operator: Linear | Conv2d = Field()

    @field_validator("weight")
    @classmethod
    def own_weight(cls, value: FloatArray) -> FloatArray:
        """Own finite weights; never optimize caller storage in place."""
        return frozen_array(value)

    @model_validator(mode="after")
    def valid_layout(self) -> Self:
        """Enforce the canonical layouts consumed by reconstruction methods."""
        match self.operator:
            case Linear():
                if self.weight.ndim != 2:
                    raise ValueError("linear weights must have rank two")
            case Conv2d(groups=groups):
                if self.weight.ndim != 4 or self.weight.shape[0] % groups:
                    raise ValueError(
                        "Conv2d requires rank-four weights and valid groups"
                    )
        match self.weight_encoding.granularity:
            case PerChannel(axis=0):
                if self.weight_encoding.scale.size != self.weight.shape[0]:
                    raise ValueError("weight encoding channel mismatch")
            case _:
                raise ValueError("reconstruction requires per-output-channel weights")
        match self.activation_encoding.granularity:
            case PerTensor():
                pass
            case PerChannel():
                raise ValueError("reconstruction requires per-tensor activations")
        return self


class Solution(BaseModel):
    """Carry exact hard weight codes and the final activation encoding."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    codes: IntArray = Field()
    activation_encoding: Encoding = Field()

    @field_validator("codes")
    @classmethod
    def own_codes(cls, value: IntArray) -> IntArray:
        """Freeze a private copy of the supported integer representation."""
        if value.dtype not in (np.dtype(np.int8), np.dtype(np.uint8)) or not value.size:
            raise ValueError("expected nonempty int8 or uint8 codes")
        owned = value.copy()
        owned.flags.writeable = False
        return owned


@runtime_checkable
class Reconstructor(Protocol):
    """Optimize one unit using replay, returning expected data/numerical failures."""

    def reconstruct(
        self, problem: Problem, replay: Replay
    ) -> Result[Solution, QuantSmithError]:
        """Consume bounded batches; propagate unexpected plugin exceptions."""
        ...
