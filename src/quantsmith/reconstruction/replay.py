"""Paired streaming execution, independent of any runtime implementation."""

from collections.abc import Callable, Iterable

from quantsmith.calibration import Evaluator, Samples
from quantsmith.domain import FloatArray
from quantsmith.reconstruction import Batch, Replay
from quantsmith.result import (
    Err,
    FailureKind,
    Ok,
    QuantSmithError,
    Result,
    failure,
    validate,
)


def paired_replay(
    reference: Evaluator,
    candidate: Evaluator,
    samples: Samples,
    tensor: str,
    layout: Callable[[FloatArray], FloatArray],
) -> Replay:
    """Bind both graph revisions to the same sample and requested input tensor."""

    def replay() -> Iterable[Result[Batch, QuantSmithError]]:
        """Keep only the current paired batch; propagate evaluator failures intact."""
        for sample in samples():
            match reference.run(sample, (tensor,)):
                case Err() as error:
                    yield error
                    return
                case Ok(original):
                    pass
            match candidate.run(sample, (tensor,)):
                case Err() as error:
                    yield error
                    return
                case Ok(current):
                    pass
            if tensor not in original or tensor not in current:
                yield failure(
                    FailureKind.INVALID_DATA,
                    "paired replay",
                    "evaluator omitted requested tensor",
                )
                return
            yield validate(
                "paired replay",
                lambda: Batch(
                    reference=layout(original[tensor]),
                    candidate=layout(current[tensor]),
                ),
            )

    return replay
