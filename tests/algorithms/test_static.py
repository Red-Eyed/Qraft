"""Static W8A8 planning from small graphs and hand-supplied statistics."""

import numpy as np
import pytest

from qraft.algorithms import Needs, Statistics
from qraft.algorithms.static import StaticW8A8
from qraft.calibration import HistogramStats, MinMaxStats, Percentile, Requirement
from qraft.domain import Graph, Node, PerChannel, PerTensor
from qraft.plan import QuantizeInput
from qraft.result import FailureKind
from tests.algorithms.cases import OperatorLayout, graph_for
from tests.outcomes import expect_error, expect_ok


@pytest.fixture
def graph(operator_layout: OperatorLayout) -> Graph:
    """Provide output channels with magnitudes 127 and 254 in every layout."""
    return graph_for(
        operator_layout, np.asarray([[-127, -254], [127, 254]], dtype=np.float32)
    )


@pytest.fixture
def stats() -> Statistics:
    """Supply a known activation interval without running a collector."""
    return Statistics(
        ranges={
            Requirement(tensor="x"): MinMaxStats(
                minimum=np.asarray(-2, dtype=np.float32),
                maximum=np.asarray(3, dtype=np.float32),
            )
        },
        histograms={},
    )


def test_static_known_parameters(
    graph: Graph, stats: Statistics, operator_layout: OperatorLayout
) -> None:
    """Check exact edge decisions and output-channel axes across all layouts."""
    algorithm = StaticW8A8()
    node = graph.nodes[0]
    before = graph.weights["w"].copy()
    assert expect_ok(algorithm.requirements(node, graph)) == Needs(
        ranges=(Requirement(tensor="x"),)
    )
    plan = expect_ok(algorithm.plan(node, graph, stats))
    match plan.operations:
        case (
            QuantizeInput(node="projection", index=0, encoding=activation),
            QuantizeInput(node="projection", index=1, encoding=weight),
        ):
            assert activation.granularity == PerTensor()
            assert float(activation.scale) == pytest.approx(5 / 255)
            assert int(activation.zero_point) == 102
            assert weight.granularity == PerChannel(
                axis=operator_layout.output_weight_axis
            )
            np.testing.assert_array_equal(weight.scale, [1, 2])
            np.testing.assert_array_equal(weight.zero_point, [0, 0])
        case _:
            pytest.fail(
                "expected activation and weight quantization on the selected node"
            )
    np.testing.assert_array_equal(graph.weights["w"], before)
    assert not graph.weights["w"].flags.writeable
    assert float(stats.ranges[Requirement(tensor="x")].minimum) == -2


def test_percentile_plan_uses_supplied_histogram(graph: Graph) -> None:
    """An 80% central interval clips known outlier bins before encoding."""
    request = Requirement(tensor="x")
    stats = Statistics(
        ranges={
            request: MinMaxStats(
                minimum=np.asarray(-100, dtype=np.float32),
                maximum=np.asarray(100, dtype=np.float32),
            )
        },
        histograms={
            request: HistogramStats(
                edges=np.asarray([-100, -2, 0, 2, 100], dtype=np.float64),
                counts=np.asarray([1, 4, 4, 1], dtype=np.int64),
            )
        },
    )
    algorithm = StaticW8A8(calibration=Percentile(percentile=80))
    assert expect_ok(algorithm.requirements(graph.nodes[0], graph)) == Needs(
        histograms=(request,)
    )
    plan = expect_ok(algorithm.plan(graph.nodes[0], graph, stats))
    match plan.operations[0]:
        case QuantizeInput(encoding=encoding):
            assert float(encoding.scale) == pytest.approx(4 / 255)
            assert int(encoding.zero_point) == 128
        case _:
            pytest.fail("expected activation quantization")


@pytest.mark.parametrize("missing", ["range", "histogram"])
def test_missing_statistics(graph: Graph, stats: Statistics, missing: str) -> None:
    """Expected incomplete inputs return a data failure rather than KeyError."""
    match missing:
        case "range":
            algorithm = StaticW8A8()
            supplied = Statistics(ranges={}, histograms={})
        case _:
            algorithm = StaticW8A8(calibration=Percentile())
            supplied = stats
    error = expect_error(
        algorithm.plan(graph.nodes[0], graph, supplied), "not collected"
    )
    assert error.kind is FailureKind.INVALID_DATA


def test_zero_weight_channel() -> None:
    """A dead output channel receives a unit scale without corrupting live channels."""
    node = Node(
        name="linear", op="MatMul", inputs=("x", "w"), outputs=("y",), attributes={}
    )
    graph = Graph(
        nodes=(node,),
        weights={"w": np.asarray([[0, -127], [0, 127]], dtype=np.float32)},
    )
    stats = Statistics(
        ranges={
            Requirement(tensor="x"): MinMaxStats(
                minimum=np.asarray(0, dtype=np.float32),
                maximum=np.asarray(0, dtype=np.float32),
            )
        },
        histograms={},
    )
    plan = expect_ok(StaticW8A8().plan(node, graph, stats))
    match plan.operations:
        case (QuantizeInput(encoding=activation), QuantizeInput(encoding=weight)):
            assert float(activation.scale) == 1
            np.testing.assert_array_equal(weight.scale, [1, 1])
        case _:
            pytest.fail("expected two quantization operations")


def test_grouped_conv_is_supported() -> None:
    """Ordinary W8A8 keeps depthwise Conv support even when smoothing is unavailable."""
    node = Node(
        name="depthwise",
        op="Conv",
        inputs=("x", "w"),
        outputs=("y",),
        attributes={"group": 2},
    )
    graph = Graph(
        nodes=(node,), weights={"w": np.asarray([[[127]], [[254]]], dtype=np.float32)}
    )
    stats = Statistics(
        ranges={
            Requirement(tensor="x"): MinMaxStats(
                minimum=np.asarray(-2, dtype=np.float32),
                maximum=np.asarray(3, dtype=np.float32),
            )
        },
        histograms={},
    )
    expect_ok(StaticW8A8().requirements(node, graph))
    plan = expect_ok(StaticW8A8().plan(node, graph, stats))
    match plan.operations[1]:
        case QuantizeInput(encoding=encoding):
            assert encoding.granularity == PerChannel(axis=0)
            np.testing.assert_array_equal(encoding.scale, [1, 2])
        case _:
            pytest.fail("expected weight quantization")
