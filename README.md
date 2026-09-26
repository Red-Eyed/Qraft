# Qraft

Qraft is a typed post-training quantization framework with explicit plans.
The functional core selects algorithms and calculates plans from graph records
and supplied statistics. The imperative shell collects calibration statistics
through an injected evaluator and replayable sample factory.

Start with the [algorithm guide](src/qraft/algorithms/README.md), which includes
a numerical SmoothQuant example and standalone unit tests. Domain contracts use
frozen Pydantic models and expected failures use `returns.result.Result`.

`just check` runs Ruff, strict Pyrefly, and all tests with pytest-xdist.
`just test-algorithms` runs the independent numerical and orchestration tests.
uv uses the pinned interpreter and locked environment automatically.
