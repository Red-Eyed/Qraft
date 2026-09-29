"""Execute calibration through injected dependencies around pure planning."""

from quantsmith.algorithms.contracts import Algorithm, Statistics
from quantsmith.algorithms.core import assemble_plan, select_algorithms
from quantsmith.calibration import Evaluator, Samples, collect, histograms
from quantsmith.domain import Graph
from quantsmith.plan import QuantizationPlan
from quantsmith.result import Err, Ok, QuantSmithError, Result
from quantsmith.rules import Rules


def build_plan(
    graph: Graph,
    evaluator: Evaluator,
    samples: Samples,
    rules: Rules[Algorithm],
    *,
    histogram_bins: int = 2048,
) -> Result[QuantizationPlan, QuantSmithError]:
    """Select algorithms, collect statistics, and assemble one graph revision's plan.

    Only collection calls the injected evaluator and replayable sample source.
    Rejected requirements stop before execution; expected failures propagate intact.
    Collection streams batches and stores bounded aggregate statistics.
    """
    match select_algorithms(graph, rules):
        case Err() as error:
            return error
        case Ok(selection):
            pass
    match collect(evaluator, samples, selection.needs.ranges):
        case Err() as error:
            return error
        case Ok(ranges):
            pass
    match histograms(
        evaluator,
        samples,
        {item: ranges[item] for item in selection.needs.histograms},
        histogram_bins,
    ):
        case Err() as error:
            return error
        case Ok(bins):
            pass
    return assemble_plan(selection, Statistics(ranges=ranges, histograms=bins))
