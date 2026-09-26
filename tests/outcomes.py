"""Test assertions that preserve the failure diagnostic when success is expected."""

import pytest

from qraft.result import Err, Ok, QraftError, Result


def expect_ok[T](outcome: Result[T, QraftError]) -> T:
    """Assert success and include the structured diagnostic on test failure."""
    match outcome:
        case Err(error):
            pytest.fail(f"{error.operation}: {error.kind}: {error.detail}")
        case Ok(value):
            return value


def expect_error[T](outcome: Result[T, QraftError], detail: str) -> QraftError:
    """Assert an expected rejection and retain its structured reason for checks."""
    match outcome:
        case Err(error):
            assert detail in error.detail
            return error
        case Ok():
            pytest.fail("expected an explicit failure")
