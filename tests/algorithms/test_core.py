"""Pure selection, statistics ownership, and plan composition contracts."""

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from qraft.algorithms import (
    Algorithm,
    Needs,
    Statistics,
    assemble_plan,
    select_algorithms,
)
from qraft.algorithms.smoothquant import SmoothQuant
from qraft.algorithms.static import StaticW8A8
from qraft.calibration import HistogramStats, MinMaxStats, Percentile, Requirement
from qraft.domain import Graph, Node
from qraft.plan import QuantizationPlan, QuantizeInput
from qraft.result import Err, FailureKind, Ok, QraftError, Result
from qraft.rules import ByName, Exclude, Rule, Rules
from tests.outcomes import expect_error, expect_ok


@pytest.fixture
def graph() -> Graph:
    """Create parallel consumers sharing both activations and constant weights."""
    nodes = tuple(
        Node(
            name=name,
            op="MatMul",
            inputs=("x", "w"),
            outputs=(name + "_out",),
            attributes={},
        )
        for name in ("first", "second", "untouched")
    )
    return Graph(
        nodes=nodes,
        weights={"w": np.asarray([[-127, -254], [127, 254]], dtype=np.float32)},
    )


@pytest.fixture
def stats() -> Statistics:
    """Supply scalar and channel statistics from known observations."""
    return Statistics(
        ranges={
            Requirement(tensor="x"): MinMaxStats(
                minimum=np.asarray(-2, dtype=np.float32),
                maximum=np.asarray(3, dtype=np.float32),
            ),
            Requirement(tensor="x", axes=(-1,)): MinMaxStats(
                minimum=np.asarray([-2, -2], dtype=np.float32),
                maximum=np.asarray([3, 3], dtype=np.float32),
            ),
        },
        histograms={},
    )


def test_selection_deduplicates_needs_and_honors_last_rule(graph: Graph) -> None:
    """Share first-pass ranges while preserving configured decisions and exclusions."""
    minmax = StaticW8A8()
    percentile = StaticW8A8(calibration=Percentile())
    rules: Rules[Algorithm] = Rules(
        default=minmax,
        overrides=(
            Rule(selector=ByName(pattern="second"), decision=percentile),
            Rule(selector=ByName(pattern="untouched"), decision=SmoothQuant()),
            Rule(
                selector=ByName(pattern="untouched"),
                decision=Exclude(reason="sensitive"),
            ),
        ),
    )
    selection = expect_ok(select_algorithms(graph, rules))
    assert selection.graph is graph
    assert tuple(item.node.name for item in selection.selected) == ("first", "second")
    assert selection.selected[0].algorithm is minmax
    assert selection.selected[1].algorithm is percentile
    assert selection.excluded == ("untouched",)
    assert selection.needs == Needs(
        ranges=(Requirement(tensor="x"),), histograms=(Requirement(tensor="x"),)
    )


def test_assembly_uses_supplied_statistics(graph: Graph, stats: Statistics) -> None:
    """Assemble real algorithm patches without an evaluator or sample source."""
    rules: Rules[Algorithm] = Rules(
        default=StaticW8A8(),
        overrides=(
            Rule(
                selector=ByName(pattern="untouched"),
                decision=Exclude(reason="sensitive"),
            ),
        ),
    )
    selection = expect_ok(select_algorithms(graph, rules))
    plan = expect_ok(assemble_plan(selection, stats))
    assert plan.excluded == ("untouched",)
    assert len(plan.operations) == 4
    match plan.operations[0]:
        case QuantizeInput(node="first", index=0, encoding=encoding):
            assert float(encoding.scale) == pytest.approx(5 / 255)
            assert int(encoding.zero_point) == 102
        case _:
            pytest.fail("expected the first consumer's activation encoding")
    assert all(operation.node in {"first", "second"} for operation in plan.operations)
    assert len(selection.selected) == 2


def test_excluded_stage_needs_no_statistics(graph: Graph) -> None:
    """An all-excluded stage is a valid pure decision requiring no execution."""
    rules: Rules[Algorithm] = Rules(default=Exclude(reason="leave floating point"))
    selection = expect_ok(select_algorithms(graph, rules))
    assert selection.needs == Needs()
    plan = expect_ok(assemble_plan(selection, Statistics(ranges={}, histograms={})))
    assert plan.operations == ()
    assert plan.excluded == ("first", "second", "untouched")


def test_assembly_rejects_missing_statistics(graph: Graph) -> None:
    """The pure entry point retains algorithm diagnostics for incomplete inputs."""
    rules: Rules[Algorithm] = Rules(default=StaticW8A8())
    selection = expect_ok(select_algorithms(graph, rules))
    error = expect_error(
        assemble_plan(selection, Statistics(ranges={}, histograms={})), "not collected"
    )
    assert error.kind is FailureKind.INVALID_DATA
    assert error.operation == "first"


def test_assembly_rejects_mixed_revision_operations(
    graph: Graph, stats: Statistics
) -> None:
    """Smoothing and quantization cannot share a plan even on different nodes."""
    rules: Rules[Algorithm] = Rules(
        default=StaticW8A8(),
        overrides=(Rule(selector=ByName(pattern="second"), decision=SmoothQuant()),),
    )
    selection = expect_ok(select_algorithms(graph, rules))
    error = expect_error(assemble_plan(selection, stats), "recalibrate")
    assert error.kind is FailureKind.CONFLICT


class Reject(BaseModel):
    """An external plugin that rejects one phase with a caller-owned diagnostic."""

    model_config = ConfigDict(frozen=True, strict=True)
    admission: bool = Field()
    error: QraftError = Field()

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QraftError]:
        """Either admit without calibration or return the original admission failure."""
        return Err(self.error) if self.admission else Ok(Needs())

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QraftError]:
        """Return the original planning failure without replacing its payload."""
        return Err(self.error)


@pytest.mark.parametrize("admission", [True, False], ids=["requirements", "planning"])
def test_plugin_failure_preserves_diagnostic(graph: Graph, admission: bool) -> None:
    """Pure selection and assembly propagate external Result failures unchanged."""
    diagnostic = QraftError(
        kind=FailureKind.UNSUPPORTED, operation="external", detail="plugin declined"
    )
    rules: Rules[Algorithm] = Rules(
        default=Reject(admission=admission, error=diagnostic)
    )
    selection = select_algorithms(graph, rules)
    outcome = (
        selection
        if admission
        else assemble_plan(expect_ok(selection), Statistics(ranges={}, histograms={}))
    )
    assert expect_error(outcome, diagnostic.detail) is diagnostic


def test_statistics_detach_mutable_mapping_inputs() -> None:
    """Changing caller dictionaries cannot change already admitted planning inputs."""
    request = Requirement(tensor="x")
    ranges = {
        request: MinMaxStats(
            minimum=np.asarray(-2, dtype=np.float32),
            maximum=np.asarray(3, dtype=np.float32),
        )
    }
    histograms = {
        request: HistogramStats(
            edges=np.asarray([-2, 0, 3], dtype=np.float64),
            counts=np.asarray([1, 1], dtype=np.int64),
        )
    }
    stats = Statistics(ranges=ranges, histograms=histograms)
    ranges.clear()
    histograms.clear()
    assert float(stats.ranges[request].minimum) == -2
    np.testing.assert_array_equal(stats.histograms[request].counts, [1, 1])
    with pytest.raises(ValueError, match="read-only"):
        stats.ranges[request].minimum[...] = 0
