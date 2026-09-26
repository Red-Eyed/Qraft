"""Report quantization decisions without hiding how examples configure Qraft."""

import json
from pathlib import Path
from typing import TypedDict, assert_never

import numpy as np

from examples._shared.schema import Coverage
from qraft.domain import Graph
from qraft.plan import QuantizationPlan, QuantizeInput, RescaleInput


class ArrayRecord(TypedDict):
    """Keep an array's shape and dtype alongside human-readable numerical values."""

    dtype: str
    shape: tuple[int, ...]
    values: list[float]


def array_record(value: object) -> ArrayRecord:
    """Serialize plan arrays; reject unexpected objects using JSON's error contract."""
    match value:
        case np.ndarray() as array:
            numeric = np.asarray(array, dtype=np.float64)
            return {
                "dtype": str(array.dtype),
                "shape": tuple(int(length) for length in array.shape),
                "values": [float(number) for number in numeric.flat],
            }
        case _:
            raise TypeError(f"unsupported plan value: {type(value).__name__}")


def save_plans(path: Path, plans: tuple[QuantizationPlan, ...]) -> None:
    """Save complete stage decisions, including scalar/channel encoding arrays."""
    path.write_text(
        json.dumps(
            [plan.model_dump() for plan in plans], default=array_record, indent=2
        )
    )


def coverage(graph: Graph, plans: tuple[QuantizationPlan, ...]) -> Coverage:
    """Summarize constant-weight scope and the actual operations in saved plans."""
    eligible = sum(
        node.op in ("Conv", "MatMul", "Gemm")
        and len(node.inputs) >= 2
        and node.inputs[1] in graph.weights
        for node in graph.nodes
    )
    quantized: list[str] = []
    transformed: list[str] = []
    for plan in plans:
        for operation in plan.operations:
            match operation:
                case QuantizeInput(node=name, index=0):
                    quantized.append(name)
                case RescaleInput(node=name):
                    transformed.append(name)
                case QuantizeInput():
                    pass
                case _:
                    assert_never(operation)
    return Coverage(
        eligible_nodes=eligible,
        quantized_nodes=tuple(quantized),
        transformed_nodes=tuple(transformed),
        excluded_nodes=plans[-1].excluded if plans else (),
    )
