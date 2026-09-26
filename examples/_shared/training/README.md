# Training, data, and language/tabular reporting helpers

`data.py` owns checksum-verified Tiny Shakespeare admission and Wine's disjoint
splits with training-only normalization. `training.py` owns bounded minibatches
and optimization. `measurement.py` retains scalar totals and capped demonstrations;
`reporting.py` writes provenance and human reports. Model architectures, export,
selectors, and quantization stages live in the named example folders.

The text/Wine datasets are deliberately finite educational datasets. These helpers
are support code, not separate runnable examples. Start at the
[model × method matrix](../../README.md).
