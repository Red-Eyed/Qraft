"""Static W8A8 as an independently composable algorithm."""

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from qraft.algorithms.contracts import Needs, Statistics
from qraft.algorithms.encoding import encode
from qraft.calibration import Calibration, MinMax, Requirement, extrema
from qraft.domain import Absent, Graph, IntegerType, Node, PerChannel, PerTensor
from qraft.domain.layout import weight_axis
from qraft.plan import QuantizationPlan, QuantizeInput
from qraft.result import FailureKind, QraftError, failure, validate


class StaticW8A8(BaseModel):
    """Quantize activation and weight edges; leave bias in floating point."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    activation_type: IntegerType = Field(default=IntegerType.UINT8)
    weight_type: IntegerType = Field(default=IntegerType.INT8)
    activation_symmetric: bool = Field(default=False)
    weight_symmetric: bool = Field(default=True)
    calibration: Calibration = Field(default_factory=MinMax)

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QraftError]:
        """Declare statistics or return an unsupported-layout outcome."""
        match weight_axis(node, graph):
            case Failure() as error:
                return error
            case _:
                pass
        request = Requirement(tensor=node.inputs[0])
        if self.calibration.needs_histogram:
            return Success(Needs(histograms=(request,)))
        return Success(Needs(ranges=(request,)))

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QraftError]:
        """Produce consumer encodings, preserving expected planning failures."""
        match weight_axis(node, graph):
            case Failure() as error:
                return error
            case _ as resolved:
                axis = resolved.unwrap()
        request = Requirement(tensor=node.inputs[0])
        if request not in stats.ranges:
            return failure(
                FailureKind.INVALID_DATA,
                node.name,
                "activation statistics were not collected",
            )
        histogram = stats.histograms.get(
            request, Absent(reason="histogram was not collected")
        )
        match self.calibration.interval(stats.ranges[request], histogram):
            case Failure() as error:
                return error
            case _ as resolved:
                interval = resolved.unwrap()
        match extrema(graph.weights[node.inputs[1]], (axis,)):
            case Failure() as error:
                return error
            case _ as resolved:
                weight_stats = resolved.unwrap()
        return validate(
            node.name,
            lambda: QuantizationPlan(
                operations=(
                    QuantizeInput(
                        node=node.name,
                        index=0,
                        encoding=encode(
                            interval,
                            self.activation_type,
                            self.activation_symmetric,
                            PerTensor(),
                        ),
                    ),
                    QuantizeInput(
                        node=node.name,
                        index=1,
                        encoding=encode(
                            weight_stats,
                            self.weight_type,
                            self.weight_symmetric,
                            PerChannel(axis=axis),
                        ),
                    ),
                )
            ),
        )
