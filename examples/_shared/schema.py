"""Shared method identities, coverage, timings, and experiment provenance."""

from enum import StrEnum
from typing import assert_never

from pydantic import BaseModel, Field

from qraft.domain import Absent


class Method(StrEnum):
    """Independently measured inference variants."""

    TORCH = "torch_fp32"
    ONNX = "onnx_fp32"
    MINMAX = "int8_minmax"
    PERCENTILE = "int8_percentile"
    SMOOTHQUANT = "int8_smoothquant"


class Latency(BaseModel):
    """Warm inference timings at batch one, excluding preprocessing and startup."""

    median_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    runs: int = Field(ge=1)


class Coverage(BaseModel):
    """Expose exactly which source operators a quantization recipe selected."""

    eligible_nodes: int = Field(ge=0)
    quantized_nodes: tuple[str, ...] = Field(default=())
    transformed_nodes: tuple[str, ...] = Field(default=())
    excluded_nodes: tuple[str, ...] = Field(default=())


class Environment(BaseModel):
    """Record runtime versions and machine context for interpreting measurements."""

    python: str = Field()
    platform: str = Field()
    machine: str = Field()
    torch: str = Field()
    torchvision: str = Field()
    onnx: str = Field()
    onnxruntime: str = Field()
    qraft: str = Field()
    stackformers: str | Absent = Field(
        default_factory=lambda: Absent(reason="not recorded by this report version")
    )


class QuantizationMethod(StrEnum):
    """Select quantization recipes separately from floating-point report baselines."""

    MINMAX = "minmax"
    PERCENTILE = "percentile"
    SMOOTHQUANT = "smoothquant"

    def variant(self) -> Method:
        """Preserve existing report identities while exposing short CLI method names."""
        match self:
            case QuantizationMethod.MINMAX:
                return Method.MINMAX
            case QuantizationMethod.PERCENTILE:
                return Method.PERCENTILE
            case QuantizationMethod.SMOOTHQUANT:
                return Method.SMOOTHQUANT
            case _:
                assert_never(self)
