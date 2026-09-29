"""Expected unsupported layouts and invalid weights remain typed algorithm failures."""

import numpy as np
import pytest

from quantsmith.algorithms import Algorithm, Statistics
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.calibration import MinMaxStats, Requirement
from quantsmith.domain import Graph, Node
from quantsmith.result import FailureKind
from tests.outcomes import expect_error


@pytest.fixture(params=[StaticW8A8(), SmoothQuant()], ids=["static", "smoothquant"])
def algorithm(request: pytest.FixtureRequest) -> Algorithm:
    """Provide both built-in algorithms through their shared typed contract."""
    match request.param:
        case StaticW8A8() | SmoothQuant() as selected:
            return selected
        case _:
            pytest.fail("unexpected algorithm fixture parameter")


@pytest.mark.parametrize(
    "problem", ["missing-weight", "bad-rank", "unsupported-operator"]
)
def test_unsupported_layout(algorithm: Algorithm, problem: str) -> None:
    """Both requirements and planning reject unsupported inputs without a backend."""
    operator = "Add" if problem == "unsupported-operator" else "MatMul"
    node = Node(
        name="linear", op=operator, inputs=("x", "w"), outputs=("y",), attributes={}
    )
    shape = (2,) if problem == "bad-rank" else (2, 2)
    weights = (
        {} if problem == "missing-weight" else {"w": np.ones(shape, dtype=np.float32)}
    )
    graph = Graph(nodes=(node,), weights=weights)
    requirements = algorithm.requirements(node, graph)
    assert expect_error(requirements, "").kind is FailureKind.UNSUPPORTED
    outcome = algorithm.plan(node, graph, Statistics(ranges={}, histograms={}))
    assert expect_error(outcome, "").kind is FailureKind.UNSUPPORTED


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_quantizable_weight(algorithm: Algorithm, value: float) -> None:
    """Require finite selected weights while allowing infinities in mask constants."""
    node = Node(
        name="linear", op="MatMul", inputs=("x", "w"), outputs=("y",), attributes={}
    )
    graph = Graph(
        nodes=(node,), weights={"w": np.full((2, 2), value, dtype=np.float32)}
    )
    stats = Statistics(
        ranges={
            Requirement(tensor="x"): MinMaxStats(
                minimum=np.asarray(-2, dtype=np.float32),
                maximum=np.asarray(3, dtype=np.float32),
            ),
            Requirement(tensor="x", axes=(-1,)): MinMaxStats(
                minimum=np.zeros(2, dtype=np.float32),
                maximum=np.ones(2, dtype=np.float32),
            ),
        },
        histograms={},
    )
    error = expect_error(algorithm.plan(node, graph, stats), "finite")
    assert error.kind is FailureKind.INVALID_DATA
