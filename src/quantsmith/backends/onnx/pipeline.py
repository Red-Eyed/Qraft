"""Application wiring for revision-separated quantization with explicit outcomes."""

from onnx import ModelProto
from pydantic import BaseModel, ConfigDict, Field

from quantsmith.algorithms.contracts import Algorithm
from quantsmith.algorithms.shell import build_plan
from quantsmith.backends.onnx import admit, describe, lower, normalize
from quantsmith.calibration import Samples
from quantsmith.plan import QuantizationPlan
from quantsmith.result import Err, Ok, QuantSmithError, Result
from quantsmith.rules import Rules
from quantsmith.runtime import OnnxEvaluator


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
) -> Result[PipelineResult, QuantSmithError]:
    """Plan each revision and return expected admission or planning failures."""
    match admit("ONNX validation", lambda: normalize(model)):
        case Err() as error:
            return error
        case Ok(current):
            pass
    plans: list[QuantizationPlan] = []
    for rules in stages:
        match describe(current):
            case Err() as error:
                return error
            case Ok(graph):
                pass
        match build_plan(
            graph, OnnxEvaluator(current), samples, rules, histogram_bins=histogram_bins
        ):
            case Err() as error:
                return error
            case Ok(plan):
                pass
        match lower(current, plan):
            case Err() as error:
                return error
            case Ok(current):
                plans.append(plan)
    return Ok(PipelineResult(model=current, plans=tuple(plans)))
