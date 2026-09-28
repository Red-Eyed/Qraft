"""Observed-extrema calibration; see README.md for intuition and references."""

from pydantic import BaseModel, ConfigDict

from qraft.calibration.contracts import HistogramStats, MinMaxStats
from qraft.domain import Absent
from qraft.result import Ok, QraftError, Result


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
    ) -> Result[MinMaxStats, QraftError]:
        """Preserve the observed interval."""
        return Ok(stats)
