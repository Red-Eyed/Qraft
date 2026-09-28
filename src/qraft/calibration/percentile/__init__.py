"""Histogram-tail clipping; see README.md for intuition and references."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from qraft.calibration.contracts import HistogramStats, MinMaxStats
from qraft.domain import Absent
from qraft.result import FailureKind, Ok, QraftError, Result, failure, validate


class Percentile(BaseModel):
    """Clip equal mass from both tails of a bounded histogram."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    percentile: float = Field(default=99.99, gt=0, le=100, allow_inf_nan=False)

    @property
    def needs_histogram(self) -> bool:
        """Require histogram counts from a replay pass."""
        return True

    def interval(
        self, stats: MinMaxStats, histogram: HistogramStats | Absent
    ) -> Result[MinMaxStats, QraftError]:
        """Choose enclosing bin edges for the requested central probability."""
        match histogram:
            case Absent(reason=reason):
                return failure(FailureKind.INVALID_DATA, "percentile", reason)
        if self.percentile == 100:
            return Ok(stats)
        cumulative = np.cumsum(histogram.counts)
        tail = (100 - self.percentile) / 200
        first = int(np.searchsorted(cumulative, cumulative[-1] * tail, side="right"))
        last = int(np.searchsorted(cumulative, cumulative[-1] * (1 - tail)))
        low = max(float(stats.minimum), float(histogram.edges[first]))
        high = min(float(stats.maximum), float(histogram.edges[last + 1]))
        return validate(
            "percentile",
            lambda: MinMaxStats(
                minimum=np.asarray(low, dtype=np.float32),
                maximum=np.asarray(high, dtype=np.float32),
            ),
        )
