"""Test assertions that preserve the failure diagnostic when success is expected."""

import pytest
from returns.result import Failure, Result

from qraft.result import QraftError


def expect_ok[T](outcome: Result[T, QraftError]) -> T:
    """Assert success and include the structured diagnostic on test failure."""
    match outcome:
        case Failure(error):
            pytest.fail(f"{error.operation}: {error.kind}: {error.detail}")
        case _:
            return outcome.unwrap()


def expect_error[T](outcome: Result[T, QraftError], detail: str) -> QraftError:
    """Assert an expected rejection and retain its structured reason for checks."""
    match outcome:
        case Failure(error):
            assert detail in error.detail
            return error
        case _:
            pytest.fail("expected an explicit failure")
