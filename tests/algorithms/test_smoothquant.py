"""SmoothQuant scales and numerical preservation without ONNX Runtime."""

import numpy as np
import pytest

from quantsmith.algorithms import Needs, Statistics
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.calibration import MinMaxStats, Requirement
from quantsmith.domain import Graph, Node
from quantsmith.plan import RescaleInput
from quantsmith.result import FailureKind
from tests.algorithms.cases import OperatorLayout, graph_for
from tests.outcomes import expect_error, expect_ok


def test_two_channel_example() -> None:
    """Activation magnitudes [4, 9] and unit weights give scales [2, 3]."""
    node = Node(
        name="projection",
        op="MatMul",
        inputs=("x", "w"),
        outputs=("y",),
        attributes={},
    )
    graph = Graph(nodes=(node,), weights={"w": np.ones((2, 1), dtype=np.float32)})
    stats = Statistics(
        ranges={
            Requirement(tensor="x", axes=(-1,)): MinMaxStats(
                minimum=np.zeros(2, dtype=np.float32),
                maximum=np.asarray([4, 9], dtype=np.float32),
            )
        },
        histograms={},
    )

    plan = expect_ok(SmoothQuant(alpha=0.5).plan(node, graph, stats))

    match plan.operations:
        case (RescaleInput(scale=scale),):
            np.testing.assert_array_equal(scale, [2, 3])
        case _:
            pytest.fail("expected one channel-rescaling operation")


@pytest.fixture
def graph(operator_layout: OperatorLayout) -> Graph:
    """Provide contraction-channel weight magnitudes one and four."""
    return graph_for(operator_layout, np.asarray([[1, -1], [4, -2]], dtype=np.float32))


@pytest.fixture
def stats(operator_layout: OperatorLayout) -> Statistics:
    """Provide known activation-channel magnitudes four and nine."""
    return Statistics(
        ranges={
            Requirement(
                tensor="x", axes=(operator_layout.activation_axis,)
            ): MinMaxStats(
                minimum=np.asarray([-4, -9], dtype=np.float32),
                maximum=np.asarray([2, 3], dtype=np.float32),
            )
        },
        histograms={},
    )


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(0.0, (1.0, 0.25)), (0.5, (2.0, 1.5)), (1.0, (4.0, 9.0))],
    ids=["weight-only", "balanced", "activation-only"],
)
def test_known_channel_scales(
    graph: Graph,
    stats: Statistics,
    operator_layout: OperatorLayout,
    alpha: float,
    expected: tuple[float, float],
) -> None:
    """Check input-channel axes and independently calculated scales in every layout."""
    algorithm = SmoothQuant(alpha=alpha)
    node = graph.nodes[0]
    before = graph.weights["w"].copy()
    assert expect_ok(algorithm.requirements(node, graph)) == Needs(
        ranges=(Requirement(tensor="x", axes=(operator_layout.activation_axis,)),)
    )
    plan = expect_ok(algorithm.plan(node, graph, stats))
    match plan.operations:
        case (
            RescaleInput(
                node="projection",
                scale=scale,
                activation_axis=activation_axis,
                weight_axis=weight_axis,
            ),
        ):
            np.testing.assert_allclose(scale, expected)
            assert activation_axis == operator_layout.activation_axis
            assert weight_axis == operator_layout.input_weight_axis
            assert not scale.flags.writeable
        case _:
            pytest.fail("expected one channel-rescaling operation")
    np.testing.assert_array_equal(graph.weights["w"], before)
    request = Requirement(tensor="x", axes=(operator_layout.activation_axis,))
    np.testing.assert_array_equal(stats.ranges[request].minimum, [-4, -9])


def test_dead_channels_and_floating_equivalence() -> None:
    """Dead channels keep identity scales; live rescaling preserves outputs."""
    node = Node(
        name="linear", op="MatMul", inputs=("x", "w"), outputs=("y",), attributes={}
    )
    weights = np.asarray([[1], [0], [1]], dtype=np.float32)
    graph = Graph(nodes=(node,), weights={"w": weights})
    stats = Statistics(
        ranges={
            Requirement(tensor="x", axes=(-1,)): MinMaxStats(
                minimum=np.zeros(3, dtype=np.float32),
                maximum=np.asarray([0, 9, 4], dtype=np.float32),
            )
        },
        histograms={},
    )
    plan = expect_ok(SmoothQuant().plan(node, graph, stats))
    match plan.operations:
        case (RescaleInput(scale=scale),):
            np.testing.assert_array_equal(scale, [1, 1, 2])
            inputs = np.asarray([[0, 9, 4]], dtype=np.float32)
            transformed_weights = weights * scale.reshape(3, 1)
            np.testing.assert_allclose((inputs / scale) @ transformed_weights, [[4]])
            np.testing.assert_array_equal(inputs @ weights, [[4]])
        case _:
            pytest.fail("expected channel rescaling")


@pytest.mark.parametrize(
    "missing", [True, False], ids=["missing", "wrong-channel-count"]
)
def test_invalid_channel_statistics(
    graph: Graph, operator_layout: OperatorLayout, missing: bool
) -> None:
    """Missing or mismatched channel statistics return precise data failures."""
    ranges = (
        {}
        if missing
        else {
            Requirement(
                tensor="x", axes=(operator_layout.activation_axis,)
            ): MinMaxStats(
                minimum=np.zeros(3, dtype=np.float32),
                maximum=np.ones(3, dtype=np.float32),
            )
        }
    )
    stats = Statistics(ranges=ranges, histograms={})
    detail = "not collected" if missing else "dimensions differ"
    error = expect_error(SmoothQuant().plan(graph.nodes[0], graph, stats), detail)
    assert error.kind is FailureKind.INVALID_DATA


def test_grouped_conv_rejection() -> None:
    """Reject grouped Conv before requesting calibration or constructing a transform."""
    node = Node(
        name="depthwise",
        op="Conv",
        inputs=("x", "w"),
        outputs=("y",),
        attributes={"group": 2},
    )
    graph = Graph(nodes=(node,), weights={"w": np.ones((2, 1, 1), dtype=np.float32)})
    algorithm = SmoothQuant()
    error = expect_error(algorithm.requirements(node, graph), "ungrouped Conv")
    assert error.kind is FailureKind.UNSUPPORTED
    error = expect_error(
        algorithm.plan(node, graph, Statistics(ranges={}, histograms={})),
        "ungrouped Conv",
    )
    assert error.kind is FailureKind.UNSUPPORTED
