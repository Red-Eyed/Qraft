"""Small domain graphs with explicit, independently specified channel layouts."""

from collections.abc import Mapping
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from quantsmith.domain import FloatArray, Graph, Node


class OperatorLayout(BaseModel):
    """Specify a supported layout and the expected activation/weight axes."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    kind: Literal["MatMul", "BatchedMatMul", "Conv", "Gemm"] = Field()
    attributes: Mapping[str, int] = Field(default_factory=dict)
    activation_axis: int = Field()
    input_weight_axis: int = Field()
    output_weight_axis: int = Field()


def graph_for(layout: OperatorLayout, matrix: FloatArray) -> Graph:
    """Store a logical input-channel × output-channel matrix in one operator layout."""
    match layout.kind:
        case "Conv":
            weights = matrix.T.reshape(2, 2, 1)
            operator = "Conv"
        case "BatchedMatMul":
            weights = np.stack((matrix, matrix))
            operator = "MatMul"
        case "Gemm" if layout.attributes["transB"]:
            weights = matrix.T
            operator = "Gemm"
        case _:
            weights = matrix
            operator = layout.kind
    node = Node(
        name="projection",
        op=operator,
        inputs=("x", "w"),
        outputs=("y",),
        attributes=layout.attributes,
    )
    return Graph(nodes=(node,), weights={"w": weights})
