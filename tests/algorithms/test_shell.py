"""Calibration orchestration tested with in-memory execution dependencies."""

from collections.abc import Iterable, Mapping

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from qraft.algorithms import Algorithm, build_plan
from qraft.algorithms.static import StaticW8A8
from qraft.calibration import Percentile
from qraft.domain import Absent, FloatArray, Graph, InputArray, Node
from qraft.plan import QuantizeInput
from qraft.result import FailureKind, QraftError
from qraft.rules import Exclude, Rules
from tests.outcomes import expect_error, expect_ok


class RecordingEvaluator(BaseModel):
    """Expose sample tensors and record requested outputs at the execution boundary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    calls: list[tuple[str, ...]] = Field(default_factory=list)
    error: QraftError | Absent = Field(
        default_factory=lambda: Absent(reason="execution succeeds")
    )

    def run(
        self, sample: Mapping[str, InputArray], outputs: tuple[str, ...]
    ) -> Result[Mapping[str, FloatArray], QraftError]:
        """Return observations or the injected failure without executing a backend."""
        self.calls.append(outputs)
        match self.error:
            case QraftError() as error:
                return Failure(error)
            case Absent():
                return Success(
                    {
                        name: np.asarray(sample[name], dtype=np.float32)
                        for name in outputs
                    }
                )


class ReplayableSource(BaseModel):
    """Record consumption while yielding fixed small batches on every pass."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    batches: tuple[FloatArray, ...] = Field()
    passes: int = Field(default=0)

    def __call__(self) -> Iterable[Mapping[str, FloatArray]]:
        """Start a fresh replay and yield one batch at a time."""
        self.passes += 1
        for batch in self.batches:
            yield {"x": batch}


@pytest.fixture
def graph() -> Graph:
    """Two consumers share the same observed activation and weight tensor."""
    nodes = tuple(
        Node(
            name=name,
            op="MatMul",
            inputs=("x", "w"),
            outputs=(name + "_out",),
            attributes={},
        )
        for name in ("first", "second")
    )
    return Graph(
        nodes=nodes,
        weights={"w": np.asarray([[-127, -254], [127, 254]], dtype=np.float32)},
    )


@pytest.fixture
def evaluator() -> RecordingEvaluator:
    """Provide isolated execution state to each orchestration test."""
    return RecordingEvaluator()


@pytest.fixture
def source() -> ReplayableSource:
    """Replay two batches spanning the known range [-2, 3]."""
    return ReplayableSource(
        batches=(
            np.asarray([[-2, 1], [0, 3]], dtype=np.float32),
            np.asarray([[1, 2]], dtype=np.float32),
        )
    )


@pytest.mark.parametrize(
    "percentile", [False, True], ids=["minmax-one-pass", "percentile-two-passes"]
)
def test_shared_collection_and_replay(
    graph: Graph,
    evaluator: RecordingEvaluator,
    source: ReplayableSource,
    percentile: bool,
) -> None:
    """Share execution across nodes and replay only for histogram calibration."""
    algorithm = StaticW8A8(calibration=Percentile()) if percentile else StaticW8A8()
    rules: Rules[Algorithm] = Rules(default=algorithm)
    plan = expect_ok(build_plan(graph, evaluator, source, rules, histogram_bins=4))
    passes = 2 if percentile else 1
    assert source.passes == passes
    assert evaluator.calls == [("x",)] * (2 * passes)
    assert len(plan.operations) == 4
    match plan.operations[0]:
        case QuantizeInput(encoding=encoding):
            assert float(encoding.scale) == pytest.approx(5 / 255)
            assert int(encoding.zero_point) == 102
        case _:
            pytest.fail("expected activation quantization")
    np.testing.assert_array_equal(source.batches[0], [[-2, 1], [0, 3]])


def test_all_excluded_skips_execution(
    graph: Graph, evaluator: RecordingEvaluator, source: ReplayableSource
) -> None:
    """No selected nodes means neither calibration nor replay consumes the source."""
    rules: Rules[Algorithm] = Rules(default=Exclude(reason="leave floating point"))
    plan = expect_ok(build_plan(graph, evaluator, source, rules))
    assert plan.excluded == ("first", "second")
    assert source.passes == 0
    assert evaluator.calls == []


def test_admission_failure_precedes_execution(
    evaluator: RecordingEvaluator, source: ReplayableSource
) -> None:
    """Unsupported constant-weight layouts stop before any shell side effects."""
    node = Node(
        name="dynamic", op="MatMul", inputs=("x", "w"), outputs=("y",), attributes={}
    )
    graph = Graph(nodes=(node,), weights={})
    rules: Rules[Algorithm] = Rules(default=StaticW8A8())
    error = expect_error(
        build_plan(graph, evaluator, source, rules), "constant weights required"
    )
    assert error.kind is FailureKind.UNSUPPORTED
    assert source.passes == 0
    assert evaluator.calls == []


def test_execution_failure_stops_collection(
    graph: Graph, source: ReplayableSource
) -> None:
    """An execution failure propagates intact without replay or further batches."""
    diagnostic = QraftError(
        kind=FailureKind.EXECUTION,
        operation="test evaluator",
        detail="execution rejected",
    )
    evaluator = RecordingEvaluator(error=diagnostic)
    rules: Rules[Algorithm] = Rules(default=StaticW8A8(calibration=Percentile()))
    outcome = build_plan(graph, evaluator, source, rules)
    assert outcome.failure() is diagnostic
    assert source.passes == 1
    assert evaluator.calls == [("x",)]


def test_empty_source_returns_empty_failure(
    graph: Graph, evaluator: RecordingEvaluator
) -> None:
    """An empty calibration source cannot silently create an uncalibrated plan."""
    source = ReplayableSource(batches=())
    rules: Rules[Algorithm] = Rules(default=StaticW8A8())
    error = expect_error(build_plan(graph, evaluator, source, rules), "no samples")
    assert error.kind is FailureKind.EMPTY
    assert source.passes == 1
    assert evaluator.calls == []
