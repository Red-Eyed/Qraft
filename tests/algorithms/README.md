# Unit-testing quantization algorithms

Start with `test_two_channel_example` in
[`test_smoothquant.py`](test_smoothquant.py). Everything needed to understand it
is inside that test: a MatMul node, unit weights, activation magnitudes `[4, 9]`,
and expected scales `[2, 3]`. It calls the actual algorithm's `.plan()` method.

Run the tests from the project root:

```sh
just test-algorithms
just test-algorithms -k two_channel_example
just test-algorithms -k gemm11
```

The equivalent direct command is:

```sh
uv run --locked pytest --confcutdir=tests/algorithms tests/algorithms -q
```

`--confcutdir` prevents loading the parent `tests/conftest.py`, which builds ONNX
integration fixtures. These tests need no Torch, ONNX Runtime execution, model
downloads, or datasets. `just check` runs them together with the integration suite.

## What is tested

| Test module | Numerical answers or contracts |
| --- | --- |
| [`test_encoding.py`](test_encoding.py) | INT8/UINT8, symmetric/asymmetric scales and zero points, positive/negative intervals, zero channels, output ownership |
| [`test_static.py`](test_static.py) | Activation and weight encodings, output-channel axes, percentile clipping, missing statistics, dead weights, grouped Conv |
| [`test_smoothquant.py`](test_smoothquant.py) | Known scales at alpha 0, 0.5, and 1; contraction axes; dead channels; floating equivalence; invalid statistics; grouped Conv rejection |
| [`test_admission.py`](test_admission.py) | Unsupported layouts, missing constant weights, and nonfinite quantizable weights |
| [`test_core.py`](test_core.py) | Rule precedence, shared requirements, exclusions, supplied-statistics assembly, conflicts, error preservation, statistics ownership |
| [`test_shell.py`](test_shell.py) | Shared execution, one-pass MinMax/two-pass Percentile, no execution after admission rejection, empty sources, execution failures |

The layout cases cover MatMul, batched MatMul, ungrouped Conv, and all four Gemm
transpose combinations. Expected axes are explicitly specified in the fixtures,
and expected numerical values are hand-calculated rather than obtained by calling
another path through the same algorithm.

## Add a test for your algorithm

1. Build a small `Graph` and constant arrays whose expected answer is easy to check.
2. Supply `Statistics` directly. Test `.requirements()` separately to check which
   tensor and retained axes the algorithm requests.
3. Call `.plan()` and assert operation kind, target edge, axis, and numerical values.
   Check expected failures through their `QraftError` category and diagnostic.

Use pytest fixtures for reusable setup and parametrization for layout variants.
Keep the first worked example self-contained. No execution fake is needed for
algorithm math; execution dependencies belong to the shell tests.

`RecordingEvaluator` and `ReplayableSource` in `test_shell.py` expose requests and
replay counts at the actual injected boundary. They do not replace the algorithm
under test. Real lowering and runtime behavior remain covered by
[`../test_pipeline.py`](../test_pipeline.py).
