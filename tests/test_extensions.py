"""Prove third-party composition and graph selectors need no core dispatch edits."""

from collections.abc import Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from returns.result import Result, Success

from qraft.algorithms import Algorithm, Needs, Statistics, build_plan
from qraft.calibration import MinMaxStats, Requirement
from qraft.domain import Graph, Node
from qraft.plan import QuantizationPlan
from qraft.result import QraftError
from qraft.rules import ByName, ByOperator, ByTensor, Exclude, Rule, Rules
from tests.outcomes import expect_ok
from tests.test_calibration import IdentityEvaluator


class ObserveOnly:
    """An external algorithm that requests shared data and chooses exclusion."""

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QraftError]:
        """Request activation data through the common collection loop."""
        return Success(Needs(ranges=(Requirement(tensor=node.inputs[0]),)))

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QraftError]:
        """Verify typed statistics are available without backend access."""
        observed: MinMaxStats = stats.ranges[Requirement(tensor=node.inputs[0])]
        assert float(observed.maximum) == 2
        return Success(QuantizationPlan(excluded=(node.name,)))


class AfterOperator(BaseModel):
    """A custom graph-pattern selector using only domain connectivity."""

    model_config = ConfigDict(frozen=True, strict=True)
    operator: str = Field()

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Select direct consumers of a chosen producer operator."""
        produced = {
            output
            for producer in graph.nodes
            if producer.op == self.operator
            for output in producer.outputs
        }
        return bool(produced.intersection(node.inputs))


def test_external_algorithm_and_rule_precedence() -> None:
    """Compose a foreign implementation and a graph-pattern selector unchanged."""
    first = Node(
        name="first", op="Identity", inputs=("x",), outputs=("hidden",), attributes={}
    )
    second = Node(
        name="second",
        op="MatMul",
        inputs=("hidden", "w"),
        outputs=("y",),
        attributes={},
    )
    graph = Graph(nodes=(first, second), weights={})
    selector = AfterOperator(operator="Identity")
    assert selector.matches(second, graph)
    assert not selector.matches(first, graph)
    assert ByTensor(name="hidden").matches(second, graph)
    assert ByOperator(operators=frozenset({"MatMul"})).matches(second, graph)
    rules: Rules[Algorithm] = Rules(
        default=Exclude(reason="outside scope"),
        overrides=(
            Rule(selector=ByName(pattern="first"), decision=ObserveOnly()),
            Rule(selector=selector, decision=Exclude(reason="sensitive")),
        ),
    )
    values = np.asarray([1, 2], dtype=np.float32)
    sample: Mapping[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]] = {
        "x": values
    }
    plan = expect_ok(
        build_plan(graph, IdentityEvaluator(), lambda: iter((sample,)), rules)
    )
    assert set(plan.excluded) == {"first", "second"}
    assert not plan.operations
    override: Rules[Algorithm] = Rules(
        default=ObserveOnly(),
        overrides=(
            Rule(selector=ByName(pattern="*"), decision=Exclude(reason="last wins")),
        ),
    )
    assert override.resolve(first, graph) == Exclude(reason="last wins")
