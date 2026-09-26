"""Explicit operations, composed and validated before graph mutation."""

from __future__ import annotations

from typing import Self, assert_never

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from returns.result import Result, Success

from qraft.domain import Encoding, FloatArray, frozen_array
from qraft.result import FailureKind, QraftError, failure


class QuantizeInput(BaseModel):
    """Quantize only one consumer edge, preserving other tensor consumers."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    node: str = Field(min_length=1)
    index: int = Field(ge=0, strict=True)
    encoding: Encoding = Field()


class RescaleInput(BaseModel):
    """Divide one activation edge and multiply its paired weight channels."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    node: str = Field(min_length=1)
    scale: FloatArray = Field()
    activation_axis: int = Field()
    weight_axis: int = Field()

    @field_validator("scale")
    @classmethod
    def freeze_scale(cls, value: FloatArray) -> FloatArray:
        """Own a positive finite channel vector."""
        scale = frozen_array(value)
        if scale.ndim != 1 or (scale <= 0).any():
            raise ValueError("rescaling requires a positive channel vector")
        return scale


type Operation = QuantizeInput | RescaleInput


class QuantizationPlan(BaseModel):
    """Hold operations for a single graph revision; reject conflicting writes."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    operations: tuple[Operation, ...] = Field(default=())
    excluded: tuple[str, ...] = Field(default=())

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Reject duplicate operations and mixed transform/calibration revisions."""
        keys: set[tuple[str, int]] = set()
        transforms: set[str] = set()
        for operation in self.operations:
            match operation:
                case QuantizeInput(node=node, index=index):
                    if index < 0 or (node, index) in keys:
                        raise ValueError("duplicate or invalid quantization edge")
                    keys.add((node, index))
                case RescaleInput(node=node):
                    if node in transforms:
                        raise ValueError("duplicate rescaling")
                    transforms.add(node)
                case _:
                    assert_never(operation)
        if transforms and keys:
            raise ValueError("lower transforms and recalibrate before quantizing")
        affected = transforms | {node for node, _ in keys}
        if affected.intersection(self.excluded):
            raise ValueError("excluded nodes cannot have plan operations")
        return self

    def then(self, other: QuantizationPlan) -> Result[QuantizationPlan, QraftError]:
        """Combine patches or return a conflict without changing either source plan."""
        try:
            return Success(
                QuantizationPlan(
                    operations=self.operations + other.operations,
                    excluded=tuple(dict.fromkeys(self.excluded + other.excluded)),
                )
            )
        except ValidationError as error:
            return failure(FailureKind.CONFLICT, "plan composition", str(error))
