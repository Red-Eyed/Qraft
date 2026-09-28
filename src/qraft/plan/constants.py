"""Owned integer replacements for consumer-local reconstructed weights."""

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qraft.domain import Encoding, IntArray, PerChannel


class QuantizeConstant(BaseModel):
    """Replace one constant edge with exact integer codes and dequantization.

    Keeping codes avoids rounding learned decisions a second time during export.
    Other consumers of the original initializer remain untouched.
    """

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    node: str = Field(min_length=1)
    index: int = Field(ge=0, strict=True)
    values: IntArray = Field()
    encoding: Encoding = Field()

    @field_validator("values")
    @classmethod
    def own_values(cls, value: IntArray) -> IntArray:
        """Reject unsupported storage and detach codes from caller-owned arrays."""
        if value.dtype not in (np.dtype(np.int8), np.dtype(np.uint8)) or not value.size:
            raise ValueError("integer codes must be nonempty int8 or uint8 data")
        owned = value.copy()
        owned.flags.writeable = False
        return owned

    @model_validator(mode="after")
    def check_layout(self) -> Self:
        """Require matching storage and a valid per-channel broadcast."""
        if self.values.dtype != self.encoding.zero_point.dtype:
            raise ValueError("codes and zero points must have identical storage")
        match self.encoding.granularity:
            case PerChannel(axis=axis):
                if not -self.values.ndim <= axis < self.values.ndim:
                    raise ValueError("constant encoding axis out of range")
                if self.values.shape[axis] != self.encoding.scale.size:
                    raise ValueError("constant encoding channel mismatch")
        return self
