"""Closed result variants for expected failures; unexpected errors remain exceptions."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import final

from pydantic import BaseModel, ConfigDict, Field, ValidationError


@final
@dataclass(frozen=True, slots=True)
class Ok[T]:
    """Carry a successful value unchanged for explicit pattern matching."""

    value: T


@final
@dataclass(frozen=True, slots=True)
class Err[E]:
    """Carry an expected error unchanged for explicit pattern matching."""

    error: E


type Result[T, E] = Ok[T] | Err[E]


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


def failure(kind: FailureKind, operation: str, detail: str) -> Err[QraftError]:
    """Build a typed failure with enough context for a CLI or report."""
    return Err(QraftError(kind=kind, operation=operation, detail=detail))


def validate[T](operation: str, call: Callable[[], T]) -> Result[T, QraftError]:
    """Translate documented validation exceptions at an admission boundary.

    Use only where invalid input or I/O failure is expected and documented.
    Other exception types propagate; this is not a wrapper for arbitrary work.
    This helper does not establish that an arbitrary callable is exception-free.
    """
    try:
        return Ok(call())
    except (ValidationError, ValueError) as error:
        return failure(FailureKind.INVALID_DATA, operation, str(error))
    except OSError as error:
        return failure(FailureKind.IO, operation, str(error))
