"""Verify rejection reasons separately from runtime numerical behavior."""

import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel, Field


class Diagnostic(BaseModel):
    """Validate the diagnostic fields used by the static contract assertions."""

    line: int = Field()
    name: str = Field()
    severity: str = Field()


class Report(BaseModel):
    """Parse Pyrefly output without leaking loose JSON into test logic."""

    errors: list[Diagnostic] = Field()


@pytest.mark.parametrize(
    ("source", "expected", "line"),
    [
        ("from qraft.domain import PerChannel\nvalue = PerChannel(axis=1)\n", "", 0),
        (
            "from returns.result import Result\n"
            "from qraft.result import QraftError\n"
            "def read(value: Result[int, QraftError]) -> None:\n"
            "    integer: int = value\n",
            "bad-assignment",
            4,
        ),
        (
            "from returns.result import Result\n"
            "from qraft.result import QraftError\n"
            "def read(value: Result[int, QraftError]) -> str:\n"
            "    return value.unwrap()\n",
            "bad-return",
            4,
        ),
        (
            "from returns.result import Result\n"
            "from qraft.result import QraftError\n"
            "def read(value: Result[int, QraftError]) -> int:\n"
            "    return value.unwrap()\n",
            "",
            0,
        ),
        (
            "from returns.result import Result, Success\n"
            "from qraft.result import QraftError\n"
            "def stringify(value: int) -> str:\n"
            "    return str(value)\n"
            "def convert(value: Result[int, QraftError]) -> Result[str, QraftError]:\n"
            "    return value.map(stringify)\n",
            "",
            0,
        ),
        (
            "from returns.result import Result, Success\n"
            "from qraft.result import QraftError\n"
            "def stringify(value: int) -> Result[str, QraftError]:\n"
            "    return Success(str(value))\n"
            "def convert(value: Result[int, QraftError]) -> Result[str, QraftError]:\n"
            "    return value.bind(stringify)\n",
            "",
            0,
        ),
        (
            "from returns.result import Result\n"
            "from qraft.result import QraftError\n"
            "def stringify(value: int) -> str:\n"
            "    return str(value)\n"
            "def convert(value: Result[int, QraftError]) -> Result[int, QraftError]:\n"
            "    return value.map(stringify)\n",
            "bad-return",
            6,
        ),
        (
            'from qraft.domain import PerChannel\nvalue = PerChannel(axis="bad")\n',
            "bad-argument-type",
            2,
        ),
        (
            "from qraft.domain import PerChannel\nvalue = PerChannel()\n",
            "missing-argument",
            2,
        ),
        (
            "from qraft.domain import PerTensor\nvalue = PerTensor(axis=1)\n",
            "unexpected-keyword",
            2,
        ),
        (
            'from qraft.config import supported\nvalue = supported("not a plugin")\n',
            "bad-argument-type",
            2,
        ),
        (
            "from qraft.domain import PerTensor, PerChannel\n"
            "from typing import assert_never\n"
            "def incomplete(value: PerTensor | PerChannel) -> None:\n"
            "    match value:\n"
            "        case PerTensor():\n"
            "            return\n"
            "        case _:\n"
            "            assert_never(value)\n",
            "bad-argument-type",
            8,
        ),
    ],
)
def test_contract(tmp_path: Path, source: str, expected: str, line: int) -> None:
    """Require exactly the intended error at the intended source location."""
    candidate = tmp_path / "contract.py"
    candidate.write_text(source)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "pyrefly",
            "check",
            "--config",
            str(root / "pyrefly.toml"),
            "--output-format",
            "json",
            str(candidate),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    report = Report.model_validate_json(result.stdout)
    errors = [
        (item.name, item.line) for item in report.errors if item.severity == "error"
    ]
    assert errors == ([(expected, line)] if expected else [])
    assert result.returncode == (1 if expected else 0)
