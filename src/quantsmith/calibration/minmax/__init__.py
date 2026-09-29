"""Observed-extrema calibration; see README.md for intuition and references."""

from pydantic import BaseModel, ConfigDict

from quantsmith.calibration.contracts import HistogramStats, MinMaxStats
from quantsmith.domain import Absent
from quantsmith.result import Ok, QuantSmithError, Result


class MinMax(BaseModel):
    """Use observed extrema without clipping."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )

    @property
    def needs_histogram(self) -> bool:
        """Use only extrema from the first pass."""
        return False

    def interval(
        self, stats: MinMaxStats, histogram: HistogramStats | Absent
    ) -> Result[MinMaxStats, QuantSmithError]:
        """Preserve the observed interval."""
        return Ok(stats)
