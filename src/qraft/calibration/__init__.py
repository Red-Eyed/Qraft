"""Replayable sample sources and bounded reusable statistics collection."""

from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, Self, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from returns.result import Failure, Result, Success

from qraft.domain import Absent, FloatArray, InputArray, frozen_array
from qraft.result import FailureKind, QraftError, failure, validate

Samples = Callable[[], Iterable[Mapping[str, InputArray]]]


class Evaluator(Protocol):
    """Execute a graph for explicitly requested floating-point tensors."""

    def run(
        self, sample: Mapping[str, InputArray], outputs: tuple[str, ...]
    ) -> Result[Mapping[str, FloatArray], QraftError]:
        """Return finite tensors or execution failure without retaining samples."""
        ...


class Requirement(BaseModel):
    """Request extrema reduced over all dimensions except retained axes."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    tensor: str = Field(min_length=1)
    axes: tuple[int, ...] = Field(default=())


class MinMaxStats(BaseModel):
    """Own finite elementwise lower and upper bounds."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    minimum: FloatArray = Field()
    maximum: FloatArray = Field()

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Validate compatible ordered extrema and freeze their storage."""
        low, high = (frozen_array(self.minimum), frozen_array(self.maximum))
        if low.shape != high.shape or np.any(low > high):
            raise ValueError("invalid extrema")
        object.__setattr__(self, "minimum", low)
        object.__setattr__(self, "maximum", high)
        return self


class HistogramStats(BaseModel):
    """Store fixed edges and bounded bin counts independently of sample count."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    edges: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = Field()
    counts: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = Field()

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Own ordered finite edges and nonnegative populated integer counts."""
        edges, counts = self.edges.copy(), self.counts.copy()
        if edges.dtype != np.float64 or counts.dtype != np.int64:
            raise ValueError("histograms require float64 edges and int64 counts")
        if edges.ndim != 1 or counts.ndim != 1 or edges.size != counts.size + 1:
            raise ValueError("invalid histogram dimensions")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("histogram edges must be finite and increasing")
        if np.any(counts < 0) or counts.sum() <= 0:
            raise ValueError("calibration replay yielded no values or invalid counts")
        edges.flags.writeable = False
        counts.flags.writeable = False
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "counts", counts)
        return self


@runtime_checkable
class Calibration(Protocol):
    """Estimate a tensor range from a shared collection of statistics."""

    @property
    def needs_histogram(self) -> bool:
        """Declare whether range estimation needs a replay pass."""
        ...

    def interval(
        self, stats: MinMaxStats, histogram: HistogramStats | Absent
    ) -> Result[MinMaxStats, QraftError]:
        """Return a finite ordered interval or a typed failure."""
        ...


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
        return Success(stats)


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
            return Success(stats)
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


def extrema(
    values: FloatArray, axes: tuple[int, ...]
) -> Result[MinMaxStats, QraftError]:
    """Reduce finite nonempty tensors or explain an invalid numerical observation."""
    if not values.size or not np.all(np.isfinite(values)):
        return failure(
            FailureKind.INVALID_DATA,
            "extrema",
            "calibration tensors must be nonempty and finite",
        )
    if any(axis < -values.ndim or axis >= values.ndim for axis in axes):
        return failure(
            FailureKind.INVALID_DATA, "extrema", "calibration axis out of range"
        )
    retained = {axis % values.ndim for axis in axes}
    reduced = tuple(axis for axis in range(values.ndim) if axis not in retained)
    return validate(
        "extrema",
        lambda: MinMaxStats(
            minimum=np.asarray(np.min(values, axis=reduced), dtype=np.float32),
            maximum=np.asarray(np.max(values, axis=reduced), dtype=np.float32),
        ),
    )


def merge_extrema(
    previous: MinMaxStats, current: MinMaxStats
) -> Result[MinMaxStats, QraftError]:
    """Merge compatible channel statistics without hiding dimension drift."""
    if previous.minimum.shape != current.minimum.shape:
        return failure(
            FailureKind.INVALID_DATA,
            "calibration",
            "retained calibration dimensions changed",
        )
    return validate(
        "calibration",
        lambda: MinMaxStats(
            minimum=np.asarray(
                np.minimum(previous.minimum, current.minimum), dtype=np.float32
            ),
            maximum=np.asarray(
                np.maximum(previous.maximum, current.maximum), dtype=np.float32
            ),
        ),
    )


def collect(
    evaluator: Evaluator, samples: Samples, requirements: tuple[Requirement, ...]
) -> Result[dict[Requirement, MinMaxStats], QraftError]:
    """Stream deduplicated observations and return expected data failures."""
    unique = tuple(dict.fromkeys(requirements))
    outputs = tuple(dict.fromkeys(item.tensor for item in unique))
    stats: dict[Requirement, MinMaxStats] = {}
    if not unique:
        return Success(stats)
    for sample in samples():
        match evaluator.run(sample, outputs):
            case Failure() as error:
                return error
            case _ as resolved:
                observed = resolved.unwrap()
        for item in unique:
            if item.tensor not in observed:
                return failure(
                    FailureKind.INVALID_DATA,
                    "calibration",
                    f"missing output {item.tensor}",
                )
            match extrema(observed[item.tensor], item.axes):
                case Failure() as error:
                    return error
                case _ as resolved:
                    current = resolved.unwrap()
            if item in stats:
                match merge_extrema(stats[item], current):
                    case Failure() as error:
                        return error
                    case _ as resolved:
                        current = resolved.unwrap()
            stats[item] = current
    if not stats:
        return failure(
            FailureKind.EMPTY, "calibration", "calibration source yielded no samples"
        )
    return Success(stats)


def histogram_edges(
    bounds: MinMaxStats, bins: int
) -> Result[np.ndarray[tuple[int, ...], np.dtype[np.float64]], QraftError]:
    """Create scalar fixed bins or report unsupported range dimensions."""
    if bounds.minimum.ndim != 0 or bins < 2:
        return failure(
            FailureKind.INVALID_DATA,
            "histogram",
            "histograms require scalar ranges and at least two bins",
        )
    low, high = float(bounds.minimum), float(bounds.maximum)
    if low == high:
        padding = max(abs(low) * 1e-6, 1e-6)
        low, high = low - padding, high + padding
    return Success(np.linspace(low, high, bins + 1, dtype=np.float64))


def histograms(
    evaluator: Evaluator,
    samples: Samples,
    ranges: Mapping[Requirement, MinMaxStats],
    bins: int,
) -> Result[dict[Requirement, HistogramStats], QraftError]:
    """Replay into fixed bins; expose invalid replay instead of throwing."""
    if bins < 2 or any(item.axes for item in ranges):
        return failure(
            FailureKind.INVALID_DATA,
            "histogram",
            "histograms require per-tensor ranges and at least two bins",
        )
    edges: dict[Requirement, np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = {}
    for item, bounds in ranges.items():
        match histogram_edges(bounds, bins):
            case Failure() as error:
                return error
            case _ as resolved:
                value = resolved.unwrap()
                edges[item] = value
    counts = {item: np.zeros(bins, dtype=np.int64) for item in ranges}
    if not ranges:
        return Success({})
    outputs = tuple(dict.fromkeys(item.tensor for item in ranges))
    for sample in samples():
        match evaluator.run(sample, outputs):
            case Failure() as error:
                return error
            case _ as resolved:
                observed = resolved.unwrap()
        for item in ranges:
            if item.tensor not in observed:
                return failure(
                    FailureKind.INVALID_DATA,
                    "histogram",
                    f"missing output {item.tensor}",
                )
            match count_histogram(observed[item.tensor], edges[item]):
                case Failure() as error:
                    return error
                case _ as resolved:
                    count = resolved.unwrap()
                    counts[item] += count
    return validate(
        "histogram",
        lambda: {
            item: HistogramStats(edges=edges[item], counts=counts[item])
            for item in ranges
        },
    )


def count_histogram(
    values: FloatArray,
    edges: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> Result[np.ndarray[tuple[int, ...], np.dtype[np.int64]], QraftError]:
    """Count one batch, preserving why a replay cannot reuse its first-pass range."""
    match extrema(values, ()):
        case Failure() as error:
            return error
        case _:
            pass
    counts, _ = np.histogram(values, bins=edges)
    if int(counts.sum()) != values.size:
        return failure(
            FailureKind.INVALID_DATA,
            "histogram",
            "replay data exceeded calibration range",
        )
    return Success(np.asarray(counts, dtype=np.int64))
