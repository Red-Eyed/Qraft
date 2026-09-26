"""Pure rule selection and plan assembly from supplied graph statistics."""

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from qraft.algorithms.contracts import Algorithm, Needs, Statistics
from qraft.domain import Graph, Node
from qraft.plan import QuantizationPlan
from qraft.result import QraftError
from qraft.rules import Exclude, Rules


class SelectedNode(BaseModel):
    """Bind one selected graph node to its configured algorithm."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    node: Node = Field()
    algorithm: Algorithm = Field()


class Selection(BaseModel):
    """Carry decisions and calibration needs bound to the original graph snapshot.

    Ranges include the first-pass bounds needed by histogram requests. Algorithms
    and selectors must honor their pure contracts; arbitrary plugins are not frozen.
    """

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    graph: Graph = Field()
    selected: tuple[SelectedNode, ...] = Field()
    excluded: tuple[str, ...] = Field()
    needs: Needs = Field()


def select_algorithms(
    graph: Graph, rules: Rules[Algorithm]
) -> Result[Selection, QraftError]:
    """Resolve rules and deduplicate needs without executing or changing the graph.

    Requirement failures preserve the plugin's diagnostic and stop selection.
    """
    selected: list[SelectedNode] = []
    excluded: list[str] = []
    needs: list[Needs] = []
    for node in graph.nodes:
        match rules.resolve(node, graph):
            case Exclude():
                excluded.append(node.name)
            case algorithm:
                match algorithm.requirements(node, graph):
                    case Failure() as error:
                        return error
                    case _ as resolved:
                        needs.append(resolved.unwrap())
                selected.append(SelectedNode(node=node, algorithm=algorithm))
    ranges = tuple(
        dict.fromkeys(item for need in needs for item in need.ranges + need.histograms)
    )
    histograms = tuple(
        dict.fromkeys(item for need in needs for item in need.histograms)
    )
    return Success(
        Selection(
            graph=graph,
            selected=tuple(selected),
            excluded=tuple(excluded),
            needs=Needs(ranges=ranges, histograms=histograms),
        )
    )


def assemble_plan(
    selection: Selection, stats: Statistics
) -> Result[QuantizationPlan, QraftError]:
    """Combine algorithm decisions using supplied statistics; perform no calibration.

    The caller supplies statistics for the selection's graph revision. Missing
    statistics, algorithm failures, and plan conflicts return the original failure.
    """
    result = QuantizationPlan(excluded=selection.excluded)
    for selected in selection.selected:
        match selected.algorithm.plan(selected.node, selection.graph, stats):
            case Failure() as error:
                return error
            case _ as resolved:
                patch = resolved.unwrap()
        match result.then(patch):
            case Failure() as error:
                return error
            case _ as resolved:
                result = resolved.unwrap()
    return Success(result)
