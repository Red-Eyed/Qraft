"""Execute calibration through injected dependencies around pure planning."""

from returns.result import Failure, Result

from qraft.algorithms.contracts import Algorithm, Statistics
from qraft.algorithms.core import assemble_plan, select_algorithms
from qraft.calibration import Evaluator, Samples, collect, histograms
from qraft.domain import Graph
from qraft.plan import QuantizationPlan
from qraft.result import QraftError
from qraft.rules import Rules


def build_plan(
    graph: Graph,
    evaluator: Evaluator,
    samples: Samples,
    rules: Rules[Algorithm],
    *,
    histogram_bins: int = 2048,
) -> Result[QuantizationPlan, QraftError]:
    """Select algorithms, collect statistics, and assemble one graph revision's plan.

    Only collection calls the injected evaluator and replayable sample source.
    Rejected requirements stop before execution; expected failures propagate intact.
    Collection streams batches and stores bounded aggregate statistics.
    """
    match select_algorithms(graph, rules):
        case Failure() as error:
            return error
        case _ as resolved:
            selection = resolved.unwrap()
    match collect(evaluator, samples, selection.needs.ranges):
        case Failure() as error:
            return error
        case _ as resolved:
            ranges = resolved.unwrap()
    match histograms(
        evaluator,
        samples,
        {item: ranges[item] for item in selection.needs.histograms},
        histogram_bins,
    ):
        case Failure() as error:
            return error
        case _ as resolved:
            bins = resolved.unwrap()
    return assemble_plan(selection, Statistics(ranges=ranges, histograms=bins))
