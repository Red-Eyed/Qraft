"""Statistics accuracy, streaming behavior, and invalid boundary handling."""

import weakref
from collections.abc import Iterable, Mapping

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quantsmith.algorithms.encoding import encode
from quantsmith.calibration import (
    MinMaxStats,
    Percentile,
    Requirement,
    collect,
    extrema,
    histograms,
)
from quantsmith.config import QuantizationConfig
from quantsmith.domain import (
    Absent,
    Encoding,
    FloatArray,
    InputArray,
    IntegerType,
    PerTensor,
)
from quantsmith.result import Ok, QuantSmithError, Result
from tests.outcomes import expect_error, expect_ok


class IdentityEvaluator(BaseModel):
    """Expose sample tensors and record requests without holding sample values."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    calls: list[tuple[str, ...]] = Field(default_factory=list)

    def run(
        self, sample: Mapping[str, InputArray], outputs: tuple[str, ...]
    ) -> Result[Mapping[str, FloatArray], QuantSmithError]:
        """Return requested tensors without retaining them."""
        self.calls.append(outputs)
        return Ok(
            {name: np.asarray(sample[name], dtype=np.float32) for name in outputs}
        )


@pytest.fixture
def evaluator() -> IdentityEvaluator:
    """Create an isolated calibration observer."""
    return IdentityEvaluator()


def test_streaming_and_deduplication(evaluator: IdentityEvaluator) -> None:
    """Retain at most the currently consumed batches, never an entire dataset."""
    references: list[weakref.ReferenceType[FloatArray]] = []

    def samples() -> Iterable[Mapping[str, FloatArray]]:
        """Generate fresh tensors and check old tensors are released."""
        for index in range(100):
            assert sum(ref() is not None for ref in references) <= 2
            value = np.full((2, 3), index, dtype=np.float32)
            references.append(weakref.ref(value))
            yield {"x": value}

    request = Requirement(tensor="x")
    stats = expect_ok(
        collect(
            evaluator, samples, (request, request, Requirement(tensor="x", axes=(1,)))
        )
    )
    assert float(stats[request].maximum) == 99
    assert len(evaluator.calls) == 100
    assert all(call == ("x",) for call in evaluator.calls)
    assert all(ref() is None for ref in references)


def test_histogram_counts_and_percentile(evaluator: IdentityEvaluator) -> None:
    """Clip rare extremes while keeping fixed-size histogram storage."""
    values = np.concatenate([np.zeros(1000), [-100, 100]]).astype(np.float32)

    def samples() -> Iterable[Mapping[str, FloatArray]]:
        """Replay a distribution with two isolated outliers."""
        yield {"x": values}

    request = Requirement(tensor="x")
    ranges = expect_ok(collect(evaluator, samples, (request,)))
    result = expect_ok(histograms(evaluator, samples, ranges, 100))
    histogram = result[request]
    assert int(histogram.counts.sum()) == values.size
    clipped = expect_ok(Percentile(percentile=99).interval(ranges[request], histogram))
    assert float(clipped.minimum) >= -2
    assert float(clipped.maximum) <= 2
    assert histogram.counts.size == 100


@pytest.mark.parametrize("value", [0.0, 1.0, -3.0])
def test_constant_ranges(value: float, evaluator: IdentityEvaluator) -> None:
    """Constant distributions produce finite encodings and valid histograms."""
    data = np.full((2, 3), value, dtype=np.float32)
    stats = expect_ok(extrema(data, ()))
    encoded = encode(stats, IntegerType.UINT8, False, PerTensor())
    assert np.isfinite(encoded.scale).all()
    assert float(encoded.scale) > 0
    request = Requirement(tensor="x")
    hist = expect_ok(
        histograms(evaluator, lambda: iter(({"x": data},)), {request: stats}, 2048)
    )
    interval = expect_ok(Percentile().interval(stats, hist[request]))
    assert float(interval.minimum) == value
    assert float(interval.maximum) == value


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_encoding(scale: float) -> None:
    """Reject invalid scales at plan construction boundaries."""
    with pytest.raises(ValueError):
        Encoding(
            scale=np.asarray(scale, dtype=np.float32),
            zero_point=np.asarray(0, dtype=np.int8),
            granularity=PerTensor(),
        )


def test_absence_and_empty_data(evaluator: IdentityEvaluator) -> None:
    """Missing histogram and empty calibration fail with explicit reasons."""
    expect_error(
        collect(evaluator, lambda: iter(()), (Requirement(tensor="x"),)), "no samples"
    )
    stats = MinMaxStats(
        minimum=np.asarray(0, dtype=np.float32), maximum=np.asarray(1, dtype=np.float32)
    )
    expect_error(
        Percentile().interval(stats, Absent(reason="not collected")), "not collected"
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"calibration": {"kind": "unknown"}}',
        '{"calibration": {"kind": "percentile", "percentile": 101}}',
        '{"smoothquant_alpha": NaN}',
        '{"unknown": true}',
    ],
)
def test_malformed_configuration(payload: str) -> None:
    """Reject malformed external configuration before constructing algorithms."""
    with pytest.raises(ValidationError):
        QuantizationConfig.model_validate_json(payload)
