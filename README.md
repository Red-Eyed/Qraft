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

## ONNX integration

`qraft.backends.onnx.pipeline.quantize` accepts an FP32 ONNX model, a factory
yielding input mappings, and ordered quantization stages. It validates and copies
the graph, executes shared CPU calibration, lowers edge-local QDQ or SmoothQuant
plans, and returns a typed result. SmoothQuant requires fresh calibration before
static W8A8. See the [backend guide](src/qraft/backends/onnx/README.md) and
[runtime guide](src/qraft/runtime/README.md) for the boundary contracts and limits.

## Run a demonstration

`just example` creates a native ONNX projection, calibrates it, applies MinMax,
Percentile, and SmoothQuant followed by MinMax, and compares held-out outputs.
It prepares output files automatically without downloading weights or data.
Start with the [projection tutorial](examples/projection/README.md).
