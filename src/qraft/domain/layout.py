"""Pure channel-layout rules shared by quantization and transformations."""

from returns.result import Failure, Result, Success

from qraft.domain import Graph, Node
from qraft.result import FailureKind, QraftError, failure


def weight_axis(node: Node, graph: Graph) -> Result[int, QraftError]:
    """Return the output-channel axis or a typed unsupported-layout failure."""
    if len(node.inputs) < 2 or node.inputs[1] not in graph.weights:
        return failure(FailureKind.UNSUPPORTED, node.name, "constant weights required")
    rank = graph.weights[node.inputs[1]].ndim
    match node.op:
        case "Conv" if rank >= 3:
            return Success(0)
        case "MatMul" if rank >= 2:
            return Success(rank - 1)
        case "Gemm" if rank == 2:
            return Success(0 if node.attributes.get("transB", 0) else 1)
        case _:
            return failure(
                FailureKind.UNSUPPORTED,
                node.name,
                f"unsupported {node.op} weight rank {rank}",
            )


def channel_axes(node: Node, graph: Graph) -> Result[tuple[int, int], QraftError]:
    """Find contraction axes, exposing unsupported layouts to the caller."""
    match weight_axis(node, graph):
        case Failure() as error:
            return error
        case _:
            pass
    match node.op:
        case "MatMul":
            return Success((-1, graph.weights[node.inputs[1]].ndim - 2))
        case "Gemm":
            activation = 0 if node.attributes.get("transA", 0) else 1
            weight = 1 if node.attributes.get("transB", 0) else 0
            return Success((activation, weight))
        case "Conv" if node.attributes.get("group", 1) == 1:
            return Success((1, 1))
        case _:
            return failure(
                FailureKind.UNSUPPORTED,
                node.name,
                "SmoothQuant requires ungrouped Conv or a supported matrix operator",
            )
