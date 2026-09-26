"""Verify defensive copies and frozen fields after typed field validation."""

from types import MappingProxyType

import numpy as np
import pytest
from pydantic import ValidationError

from qraft.algorithms.contracts import Statistics
from qraft.calibration import HistogramStats, MinMaxStats, Requirement
from qraft.domain import Encoding, FloatArray, Graph, Node, PerChannel
from qraft.plan import RescaleInput


@pytest.fixture
def values() -> FloatArray:
    """Provide caller-owned storage shared across several validated records."""
    return np.asarray([1, 2], dtype=np.float32)


def test_float_array_ownership(values: FloatArray) -> None:
    """Construction copies arrays without freezing or retaining caller storage."""
    stats = MinMaxStats(minimum=values, maximum=values)
    rescaling = RescaleInput(node="n", scale=values, activation_axis=0, weight_axis=0)
    arrays = (stats.minimum, stats.maximum, rescaling.scale)
    values[:] = 9
    for array in arrays:
        np.testing.assert_array_equal(array, [1, 2])
        assert not np.shares_memory(array, values)
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 3


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_encoding_ownership(
    values: FloatArray, dtype: type[np.int8] | type[np.uint8]
) -> None:
    """Both integer formats retain their dtype and detach scale and zero storage."""
    zero = np.zeros(2, dtype=dtype)
    encoding = Encoding(scale=values, zero_point=zero, granularity=PerChannel(axis=0))
    values[:] = 9
    zero[:] = 7
    np.testing.assert_array_equal(encoding.scale, [1, 2])
    np.testing.assert_array_equal(encoding.zero_point, [0, 0])
    assert encoding.zero_point.dtype == dtype
    assert not encoding.scale.flags.writeable
    assert not encoding.zero_point.flags.writeable
    assert values.flags.writeable and zero.flags.writeable


def test_histogram_ownership() -> None:
    """Histogram edges and counts are independently owned read-only arrays."""
    edges = np.asarray([0, 1, 2], dtype=np.float64)
    counts = np.asarray([3, 4], dtype=np.int64)
    histogram = HistogramStats(edges=edges, counts=counts)
    edges[:] = 9
    counts[:] = 9
    np.testing.assert_array_equal(histogram.edges, [0, 1, 2])
    np.testing.assert_array_equal(histogram.counts, [3, 4])
    assert not histogram.edges.flags.writeable
    assert not histogram.counts.flags.writeable


def test_mapping_ownership(values: FloatArray) -> None:
    """Caller mapping changes cannot alter graph or statistics snapshots."""
    attributes: dict[str, int | float | tuple[int, ...]] = {"axis": 1}
    node = Node(
        name="n", op="Identity", inputs=("x",), outputs=("y",), attributes=attributes
    )
    weights = {"w": values}
    graph = Graph(nodes=(node,), weights=weights)
    request = Requirement(tensor="x")
    ranges = {request: MinMaxStats(minimum=values, maximum=values)}
    histograms = {
        request: HistogramStats(
            edges=np.asarray([0, 1], dtype=np.float64),
            counts=np.asarray([1], dtype=np.int64),
        )
    }
    stats = Statistics(ranges=ranges, histograms=histograms)
    attributes.clear()
    weights.clear()
    ranges.clear()
    histograms.clear()
    assert node.attributes == {"axis": 1}
    assert tuple(graph.weights) == ("w",)
    assert tuple(stats.ranges) == (request,)
    assert tuple(stats.histograms) == (request,)
    for mapping in (node.attributes, graph.weights, stats.ranges, stats.histograms):
        assert isinstance(mapping, MappingProxyType)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [([2], [1]), ([0, 1], [2]), ([], []), ([float("nan")], [1])],
)
def test_invalid_extrema(minimum: list[float], maximum: list[float]) -> None:
    """Independent normalization must retain shape, finiteness, and order checks."""
    with pytest.raises(ValidationError):
        MinMaxStats(
            minimum=np.asarray(minimum, dtype=np.float32),
            maximum=np.asarray(maximum, dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("edges", "counts"),
    [([0, 1, 2], [1]), ([1, 0], [1]), ([0, float("inf")], [1])],
)
def test_invalid_histogram(edges: list[float], counts: list[int]) -> None:
    """Cross-field lengths and per-field ordering remain enforced."""
    with pytest.raises(ValidationError):
        HistogramStats(
            edges=np.asarray(edges, dtype=np.float64),
            counts=np.asarray(counts, dtype=np.int64),
        )
