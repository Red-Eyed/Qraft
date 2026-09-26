# Example support helpers

The readable workflows live in each named example's `main.py`. This package keeps
supporting work out of those reading paths:

- `training/data.py`: cached text, finite tabular data, and disjoint splits.
- `training/training.py`: minibatch optimization and validated eager inference.
- `vision/data.py`: image downloads, file selection, and one-image decoding.
- `vision/models.py`: preprocessing and Torch/ORT output validation.
- `training/measurement.py` and `vision/measurement.py`: bounded metric totals and
  prediction galleries/continuations.
- `*/reporting.py`, `plans.py`, and `environment.py`: saved provenance, NumPy-aware
  stage-plan JSON, and portable HTML/Markdown reports.
- `cli.py`: stderr logs and human/JSON final presentation.

There is no shared model-suite runner. Entry points show their model loading,
Torch export, calibration mappings, and Qraft calls explicitly. Pure numerical
metrics are separated from cache/disk/runtime work; input arrays stay native.

[Browse the runnable model × method matrix](../README.md).
