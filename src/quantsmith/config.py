"""Validated external configuration and built-in construction at the API edge."""

from typing import Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field

from quantsmith.algorithms import Algorithm
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.calibration import MinMax, Percentile
from quantsmith.domain import IntegerType
from quantsmith.rules import ByOperator, Exclude, Rule, Rules


class MinMaxConfig(BaseModel):
    """Choose untrimmed observed extrema."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["minmax"] = Field(default="minmax")


class PercentileConfig(BaseModel):
    """Choose a validated central percentile interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["percentile"] = Field(default="percentile")
    percentile: float = Field(default=99.99, gt=0, le=100, allow_inf_nan=False)


class QuantizationConfig(BaseModel):
    """Parse built-in defaults without restricting the open Python plugin API."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    activation_type: IntegerType = Field(default=IntegerType.UINT8)
    weight_type: IntegerType = Field(default=IntegerType.INT8)
    activation_symmetric: bool = Field(default=False, strict=True)
    weight_symmetric: bool = Field(default=True, strict=True)
    calibration: MinMaxConfig | PercentileConfig = Field(
        default_factory=MinMaxConfig, discriminator="kind"
    )
    histogram_bins: int = Field(default=2048, ge=2, strict=True)
    smoothquant_alpha: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    smoothquant: bool = Field(default=False, strict=True)

    def stages(self) -> tuple[Rules[Algorithm], ...]:
        """Construct built-ins only at the edge, leaving plugin dispatch open."""
        match self.calibration:
            case MinMaxConfig():
                calibration = MinMax()
            case PercentileConfig(percentile=percentile):
                calibration = Percentile(percentile=percentile)
            case _:
                assert_never(self.calibration)
        static = StaticW8A8(
            activation_type=self.activation_type,
            weight_type=self.weight_type,
            activation_symmetric=self.activation_symmetric,
            weight_symmetric=self.weight_symmetric,
            calibration=calibration,
        )
        stages = (supported(static),)
        if self.smoothquant:
            return (supported(SmoothQuant(alpha=self.smoothquant_alpha)),) + stages
        return stages


def supported(algorithm: Algorithm) -> Rules[Algorithm]:
    """Apply an algorithm to supported operators and explicitly exclude others."""
    return Rules(
        default=Exclude(reason="operator outside this stage's scope"),
        overrides=(
            Rule(
                selector=ByOperator(operators=frozenset({"Conv", "MatMul", "Gemm"})),
                decision=algorithm,
            ),
        ),
    )
