"""Application wiring for revision-separated quantization with explicit outcomes."""

from onnx import ModelProto
from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from qraft.algorithms.contracts import Algorithm
from qraft.algorithms.shell import build_plan
from qraft.backends.onnx import admit, describe, lower, normalize
from qraft.calibration import Samples
from qraft.plan import QuantizationPlan
from qraft.result import QraftError
from qraft.rules import Rules
from qraft.runtime import OnnxEvaluator


class PipelineResult(BaseModel):
    """Expose the final model and each revision's inspectable decisions."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    model: ModelProto = Field()
    plans: tuple[QuantizationPlan, ...] = Field()


def quantize(
    model: ModelProto,
    samples: Samples,
    stages: tuple[Rules[Algorithm], ...],
    *,
    histogram_bins: int = 2048,
) -> Result[PipelineResult, QraftError]:
    """Plan each revision and return expected admission or planning failures."""
    match admit("ONNX validation", lambda: normalize(model)):
        case Failure() as error:
            return error
        case _ as resolved:
            current = resolved.unwrap()
    plans: list[QuantizationPlan] = []
    for rules in stages:
        match describe(current):
            case Failure() as error:
                return error
            case _ as resolved:
                graph = resolved.unwrap()
        match build_plan(
            graph, OnnxEvaluator(current), samples, rules, histogram_bins=histogram_bins
        ):
            case Failure() as error:
                return error
            case _ as resolved:
                plan = resolved.unwrap()
        match lower(current, plan):
            case Failure() as error:
                return error
            case _ as resolved:
                current = resolved.unwrap()
                plans.append(plan)
    return Success(PipelineResult(model=current, plans=tuple(plans)))
