"""Structured Qraft diagnostics carried by the returns package's Result."""

from collections.abc import Callable
from enum import StrEnum
from typing import Never

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from returns.result import Failure, Result, Success


class FailureKind(StrEnum):
    """Stable categories for decisions a caller can route without parsing text."""

    INVALID_DATA = "invalid_data"
    UNSUPPORTED = "unsupported"
    CONFLICT = "conflict"
    EXECUTION = "execution"
    IO = "io"
    EMPTY = "empty"


class QraftError(BaseModel):
    """Preserve the operation, category, and diagnostic for an expected failure."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    kind: FailureKind = Field()
    operation: str = Field(min_length=1)
    detail: str = Field(min_length=1)


def failure(
    kind: FailureKind, operation: str, detail: str
) -> Result[Never, QraftError]:
    """Build a typed failure with enough context for a CLI or report."""
    return Failure(QraftError(kind=kind, operation=operation, detail=detail))


def validate[T](operation: str, call: Callable[[], T]) -> Result[T, QraftError]:
    """Translate documented validation exceptions at an admission boundary.

    Programming errors, interruption, and unexpected dependency exceptions propagate.
    This helper does not establish that an arbitrary callable is exception-free.
    """
    try:
        return Success(call())
    except (ValidationError, ValueError) as error:
        return failure(FailureKind.INVALID_DATA, operation, str(error))
    except OSError as error:
        return failure(FailureKind.IO, operation, str(error))
