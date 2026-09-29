"""Edge-local SmoothQuant; see README.md for the paper and a worked example."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from quantsmith.algorithms.contracts import Needs, Statistics
from quantsmith.calibration import Requirement, extrema
from quantsmith.domain import Graph, Node
from quantsmith.domain.layout import channel_axes
from quantsmith.plan import QuantizationPlan, RescaleInput
from quantsmith.result import (
    Err,
    FailureKind,
    Ok,
    QuantSmithError,
    Result,
    failure,
    validate,
)


class SmoothQuant(BaseModel):
    """Balance activation and weight channels before recollecting statistics."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    alpha: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Request contraction-channel extrema or explain an unsupported layout."""
        match channel_axes(node, graph):
            case Err() as error:
                return error
            case Ok(axes):
                activation_axis, _ = axes
                return Ok(
                    Needs(
                        ranges=(
                            Requirement(tensor=node.inputs[0], axes=(activation_axis,)),
                        )
                    )
                )

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QuantSmithError]:
        """Use identity scaling for dead channels and expose incompatible statistics."""
        match channel_axes(node, graph):
            case Err() as error:
                return error
            case Ok(axes):
                activation_axis, axis = axes
        request = Requirement(tensor=node.inputs[0], axes=(activation_axis,))
        if request not in stats.ranges:
            return failure(
                FailureKind.INVALID_DATA,
                node.name,
                "channel statistics were not collected",
            )
        observed = stats.ranges[request]
        match extrema(graph.weights[node.inputs[1]], (axis,)):
            case Err() as error:
                return error
            case Ok(weights):
                pass
        activation = np.maximum(
            np.abs(observed.minimum), np.abs(observed.maximum)
        ).astype(np.float64)
        weight = np.maximum(np.abs(weights.minimum), np.abs(weights.maximum)).astype(
            np.float64
        )
        if activation.shape != weight.shape:
            return failure(
                FailureKind.INVALID_DATA,
                node.name,
                "activation and weight channel dimensions differ",
            )
        active = (activation > 0) & (weight > 0)
        scale = np.ones_like(activation)
        scale[active] = activation[active] ** self.alpha / weight[active] ** (
            1 - self.alpha
        )
        return validate(
            node.name,
            lambda: QuantizationPlan(
                operations=(
                    RescaleInput(
                        node=node.name,
                        scale=scale.astype(np.float32),
                        activation_axis=activation_axis,
                        weight_axis=axis,
                    ),
                )
            ),
        )
