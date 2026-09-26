# Image data and evaluation helpers

`data.py` caches Imagenette and selects bounded train/val file metadata. `models.py`
validates preprocessing and external Torch/ORT outputs. `measurement.py` streams
images into scalar metrics and a capped gallery; `reporting.py` renders measured
results. Concrete checkpoint loading, export, and quantization are visible in
`resnet18/main.py`, `mobilenet_v2/main.py`, and `efficientnet_b0/main.py`.

These helpers are not separate runnable examples. Start at the
[model × method matrix](../../README.md).
