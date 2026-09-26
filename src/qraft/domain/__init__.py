"""Backend-independent graph records and numerical contracts."""

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Self, assert_never

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

FloatArray = NDArray[np.float32]
type InputArray = FloatArray | NDArray[np.int64]
IntArray = NDArray[np.int8] | NDArray[np.uint8]


class Absent(BaseModel):
    """Preserve why domain information is unavailable."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    reason: str = Field()


class PerTensor(BaseModel):
    """Use one encoding for an entire tensor."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )


class PerChannel(BaseModel):
    """Use one encoding per coordinate on the specified axis."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    axis: int = Field()


type Granularity = PerTensor | PerChannel


class IntegerType(Enum):
    """Supported storage types; consumers must handle each member."""

    INT8 = "int8"
    UINT8 = "uint8"

    @property
    def bounds(self) -> tuple[int, int]:
        """Return the representable closed integer interval."""
        match self:
            case IntegerType.INT8:
                return (-128, 127)
            case IntegerType.UINT8:
                return (0, 255)
            case _:
                assert_never(self)


class Node(BaseModel):
    """Describe graph connectivity without backend objects."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    name: str = Field(min_length=1)
    op: str = Field(min_length=1)
    inputs: tuple[str, ...] = Field()
    outputs: tuple[str, ...] = Field()
    attributes: Mapping[str, int | float | tuple[int, ...]] = Field()

    @model_validator(mode="after")
    def freeze_attributes(self) -> Self:
        """Detach the attribute mapping so callers cannot mutate this snapshot."""
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        return self


class Graph(BaseModel):
    """Expose a graph snapshot and owned read-only floating-point constants."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    nodes: tuple[Node, ...] = Field()
    weights: Mapping[str, FloatArray] = Field()

    @model_validator(mode="after")
    def freeze_weights(self) -> Self:
        """Own immutable constants and require unique node identities."""
        if len({node.name for node in self.nodes}) != len(self.nodes):
            raise ValueError("graph node identities must be unique")
        weights = {name: frozen_constant(value) for name, value in self.weights.items()}
        object.__setattr__(self, "weights", MappingProxyType(weights))
        return self


class Encoding(BaseModel):
    """Own finite positive scales and shape-matched integer zero points."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    scale: FloatArray = Field()
    zero_point: IntArray = Field()
    granularity: Granularity = Field()

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Reject invalid numerical encodings and detach mutable aliases."""
        scale = np.array(self.scale, dtype=np.float32, copy=True)
        zero = np.array(self.zero_point, copy=True)
        if zero.dtype not in (np.dtype(np.int8), np.dtype(np.uint8)):
            raise ValueError("zero point must have int8 or uint8 storage")
        if scale.shape != zero.shape or not np.all(np.isfinite(scale)):
            raise ValueError("encoding shapes must match and scales must be finite")
        if np.any(scale <= 0):
            raise ValueError("scales must be positive")
        match self.granularity:
            case PerTensor():
                if scale.ndim != 0:
                    raise ValueError("per-tensor encoding requires scalars")
            case PerChannel():
                if scale.ndim != 1 or scale.size == 0:
                    raise ValueError("per-channel encoding requires a nonempty vector")
            case _:
                assert_never(self.granularity)
        scale.flags.writeable = False
        zero.flags.writeable = False
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "zero_point", zero)
        return self


def frozen_constant(value: FloatArray) -> FloatArray:
    """Own graph constants, including IEEE infinities in additive attention masks.

    Finiteness is enforced when a constant is used as a quantizable weight or a
    numerical statistic. Unselected constants preserve the source graph exactly.
    """
    if value.dtype != np.dtype(np.float32):
        raise ValueError("expected float32 data")
    result = np.array(value, dtype=np.float32, copy=True)
    result.flags.writeable = False
    return result


def frozen_array(value: FloatArray) -> FloatArray:
    """Copy finite nonempty data into an owned read-only float32 array."""
    if value.dtype != np.dtype(np.float32):
        raise ValueError("expected float32 data")
    result = np.array(value, dtype=np.float32, copy=True)
    if not result.size or not np.all(np.isfinite(result)):
        raise ValueError("expected nonempty finite data")
    result.flags.writeable = False
    return result
