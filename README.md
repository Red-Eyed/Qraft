# Qraft

**Post-training quantization you can inspect, tune, and measure.**

Turn an FP32 ONNX model into a W8A8 QDQ graph using representative calibration
inputs, without retraining. Compare MinMax, Percentile, and SmoothQuant, choose
different methods for different layers, and keep sensitive layers in floating point.

Qraft gives you the quantized model **and the decisions behind it**: selected
layers, scales, zero points, exclusions, and channel transforms. Use those plans
to understand a quality regression and decide what to change next.

Think **“craft, with Q for quantization.”**
The informal **QRAFT** mnemonic stands for **Quantization, Rules, Algorithms,
Functional core, and Transformations**.

## Try a complete workflow

From this checkout, run:

```sh
just example
```

This builds a small ONNX projection, compares all three quantization methods,
and saves models, plans, and held-out error measurements to
`artifacts/projection/index.html`. It needs no model or dataset downloads;
uv provisions the pinned Python interpreter and locked dependencies automatically.

For a Torch model, start with a causal Transformer:

```sh
just example transformer --methods minmax,percentile
just example windowed_transformer --methods smoothquant
```

These examples train a small language model, export it to ONNX, calibrate it,
and compare quantized predictions and perplexity against Torch and ONNX FP32.
Each folder has a README and an explicit `main.py` you can adapt to your own task.

| Your task | Start here |
| --- | --- |
| Global causal attention | [Transformer](examples/transformer/README.md) |
| Local windowed attention | [Windowed Transformer](examples/windowed_transformer/README.md) |
| Tabular classification | [Wine MLP](examples/wine_mlp/README.md) |
| Pretrained image classification | [ResNet-18](examples/resnet18/README.md), [MobileNetV2](examples/mobilenet_v2/README.md), [EfficientNet-B0](examples/efficientnet_b0/README.md) |
| Recurrent sequence model | [Character RNN](examples/character_rnn/README.md) |
| Smallest API walkthrough | [ONNX projection](examples/projection/README.md) |

Every model supports all three methods. Browse the
**[models × quantization methods matrix](examples/README.md)** for commands
and explanations, or use `just example <model> --help` to adjust budgets.

## Quantize your model

For an ONNX model with a float32 input named `features`, and representative
calibration rows saved in `calibration.npy`:

```python
from collections.abc import Iterator, Mapping

import numpy as np
import onnx

from qraft.backends.onnx.pipeline import quantize
from qraft.config import QuantizationConfig
from qraft.domain import InputArray
from qraft.result import Err, Ok

features = np.load("calibration.npy", mmap_mode="r")


def samples() -> Iterator[Mapping[str, InputArray]]:
    """Replay representative inputs for every calibration pass."""
    for index in range(len(features)):
        yield {"features": np.asarray(features[index : index + 1], dtype=np.float32)}


config = QuantizationConfig(smoothquant=True)
outcome = quantize(onnx.load("model.onnx"), samples, config.stages())
match outcome:
    case Err(error):
        print(error.kind, error.operation, error.detail)
    case Ok(result):
        onnx.save(result.model, "model_w8a8.onnx")
        # result.plans exposes every stage's quantization decisions.
```

Use the exact input names and shapes from your exported model. Token inputs
remain int64; feature inputs remain float32. The factory must replay the same
representative inputs on each pass, with held-out data reserved for evaluation.

Qraft 0.6 uses its own frozen `Ok[T] | Err[E]` union. Import `Ok`, `Err`, and
`Result` from `qraft.result` and match the variants directly. Code and plugins
using the previous external containers must update their imports and return
values; there are no `unwrap`, `map`, or `bind` methods.
Use results for expected failures callers can handle. Operations expected to
succeed can return normal values and raise on failure; unexpected errors and
broken invariants remain exceptions.

Start with `QuantizationConfig()` for MinMax. Enable `smoothquant=True` to balance
channels before fresh W8A8 calibration, or use `PercentileConfig` to clip outliers.
For per-layer control, compose [rules](src/qraft/rules/README.md) with
`StaticW8A8`, `SmoothQuant`, and `Exclude`; the Transformer examples show concrete
selectors that quantize constant-weight projections while keeping dynamic
attention products in floating point.

## Measure the tradeoff

Calibration choices can change task quality substantially. In the
[EfficientNet-B0 demonstration](examples/efficientnet_b0/README.md), a 256-image
subset measured **75.00% FP32 top-1**, **44.92% with MinMax**, and **69.53% with
Percentile**. This is a subset experiment, not an accuracy guarantee; the
examples expose failures as well as successful approximations.

Reports include held-out task metrics or output errors, predictions, saved
graphs, and stage plans. Compare methods on your own data before choosing one.

## Current scope

Qraft supports FP32 ONNX graphs, constant-weight Conv/MatMul/Gemm, INT8 or UINT8
encodings, per-tensor activations, and per-output-channel weights. Calibration
streams samples and keeps bounded statistics; the current runtime uses CPU
ONNX Runtime. SmoothQuant supports ungrouped Conv and matrix projections.

QDQ insertion does not guarantee faster inference or smaller files: weight
initializers, biases, and terminal outputs remain floating point. GPU calibration,
weight-only quantization, MSE calibration, AdaRound, and SeqMSE are future work.

## Go deeper

- [Algorithm guide and numerical examples](src/qraft/algorithms/README.md)
- [ONNX graph and lowering contracts](src/qraft/backends/onnx/README.md)
- [Calibration and replay requirements](src/qraft/calibration/README.md)
- [Runtime evaluation](src/qraft/runtime/README.md)
- [Algorithm unit tests](tests/algorithms/README.md)

`just check` runs Ruff, strict Pyrefly, and the full test suite.
`just test` uses pytest-xdist; `just test-algorithms` runs the independent
numerical tests. `just wheel` builds the package.
