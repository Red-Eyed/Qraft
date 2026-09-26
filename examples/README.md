# Models × quantization methods

Each folder contains its own README and a concrete `main.py`. Choose a model
and method below; `just example <model>` runs all three methods.

| Model | MinMax | Percentile | SmoothQuant → MinMax |
| --- | --- | --- | --- |
| [transformer](transformer/README.md) | `just example transformer --methods minmax` | `just example transformer --methods percentile` | `just example transformer --methods smoothquant` |
| [windowed_transformer](windowed_transformer/README.md) | `just example windowed_transformer --methods minmax` | `just example windowed_transformer --methods percentile` | `just example windowed_transformer --methods smoothquant` |
| [wine_mlp](wine_mlp/README.md) | `just example wine_mlp --methods minmax` | `just example wine_mlp --methods percentile` | `just example wine_mlp --methods smoothquant` |
| [projection](projection/README.md) | `just example projection --methods minmax` | `just example projection --methods percentile` | `just example projection --methods smoothquant` |
