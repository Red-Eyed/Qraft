# Models × quantization methods

Choose a row and a method. Each model folder has a README and a concrete
[`main.py`](transformer/main.py) that shows model setup, ONNX export, replayable
calibration inputs, Qraft configuration, and held-out comparison. The code is
intentionally explicit: you can read one example without following a shared runner.

| Model / task | MinMax | Percentile | SmoothQuant → MinMax |
| --- | --- | --- | --- |
| [Global causal Transformer](transformer/README.md) | [Read & run](transformer/README.md#minmax) | [Read & run](transformer/README.md#percentile) | [Read & run](transformer/README.md#smoothquant) |
| [Windowed causal Transformer](windowed_transformer/README.md) | [Read & run](windowed_transformer/README.md#minmax) | [Read & run](windowed_transformer/README.md#percentile) | [Read & run](windowed_transformer/README.md#smoothquant) |
| [Wine MLP](wine_mlp/README.md) | [Read & run](wine_mlp/README.md#minmax) | [Read & run](wine_mlp/README.md#percentile) | [Read & run](wine_mlp/README.md#smoothquant) |
| [ResNet-18](resnet18/README.md) | [Read & run](resnet18/README.md#minmax) | [Read & run](resnet18/README.md#percentile) | [Read & run](resnet18/README.md#smoothquant) |
| [MobileNetV2](mobilenet_v2/README.md) | [Read & run](mobilenet_v2/README.md#minmax) | [Read & run](mobilenet_v2/README.md#percentile) | [Read & run](mobilenet_v2/README.md#smoothquant) |
| [EfficientNet-B0](efficientnet_b0/README.md) | [Read & run](efficientnet_b0/README.md#minmax) | [Read & run](efficientnet_b0/README.md#percentile) | [Read & run](efficientnet_b0/README.md#smoothquant) |
| [Character RNN](character_rnn/README.md) | [Read & run](character_rnn/README.md#minmax) | [Read & run](character_rnn/README.md#percentile) | [Read & run](character_rnn/README.md#smoothquant) |
| [Small projection](projection/README.md) | [Read & run](projection/README.md#minmax) | [Read & run](projection/README.md#percentile) | [Read & run](projection/README.md#smoothquant) |

The Transformer examples are the primary reading path. Start with
[the projection](projection/README.md) if you want the smallest Qraft API example.
The RNN is a secondary demonstration.

## Read and run one example

Open its README, then read `run()` near the top of `main.py`. The same file contains
the exporter, calibration factory, and quantization configurations. A local
`model.py` holds the trained model's architecture; `schema.py` holds CLI controls.
Downloads, minibatch training, metric accumulation, and HTML rendering are support
helpers in [`_shared`](_shared/README.md).

From the project root:

```sh
just example transformer --methods percentile
just example transformer --methods minmax,percentile
just example transformer
```

The last command compares all three methods. The direct command is:

```sh
uv run --locked --group examples -m examples.transformer.main --methods percentile
```

The entry point prepares its cache, directories, exports, and reports automatically.
`--help` lists budgets, `--quiet` disables progress, and `--json` prints structured
results while logs stay on stderr. Model identity belongs to the chosen folder;
method selection changes the matrix columns that run.

## What the method columns mean

- **MinMax:** static W8A8 using the full observed activation interval.
- **Percentile:** static W8A8 after histogram-based clipping of activation outliers.
- **SmoothQuant → MinMax:** balance activation/weight channels, lower that floating
  transform, then collect fresh statistics and apply static W8A8.

Every selected method starts from the same FP32 model. The seven Torch examples
check FP32 ONNX parity first and compare against both Torch and ONNX baselines.
The projection is constructed directly in ONNX and compares against its FP32 graph.
Calibration and held-out data remain separate.

## Input contracts

| Example | ONNX input after preprocessing |
| --- | --- |
| [Global causal Transformer](transformer/README.md) | `tokens`: int64 `[1, 32]` |
| [Windowed causal Transformer](windowed_transformer/README.md) | `tokens`: int64 `[1, 32]` |
| [Wine MLP](wine_mlp/README.md) | `features`: float32 `[1, 13]` |
| [ResNet-18](resnet18/README.md) | `images`: float32 `[1, 3, 224, 224]` |
| [MobileNetV2](mobilenet_v2/README.md) | `images`: float32 `[1, 3, 224, 224]` |
| [EfficientNet-B0](efficientnet_b0/README.md) | `images`: float32 `[1, 3, 224, 224]` |
| [Character RNN](character_rnn/README.md) | `tokens`: int64 `[1, 32]` |
| [Small projection](projection/README.md) | `x`: float32 `[16, 8]` |

Text context length is configurable before fixed-shape export. Image transforms
come from the exact pretrained checkpoint. Qraft takes native NumPy input arrays;
the factory must yield the exact ONNX input names and replay the same inputs on
every calibration pass.

## Outputs and interpretation

Each example writes `artifacts/<example>/index.html`, `results.json`, and
`summary.md`, plus its FP32/quantized ONNX graphs and stage plans. Reports show
actual held-out predictions or output errors, not just a successful export.
The trained language models are small educational models, not pretrained LLMs.
QDQ insertion does not establish artifact compression, kernel fusion, or speedup;
compare measured task quality and latency in the report.
