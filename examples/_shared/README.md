# Example support helpers

The concrete workflow lives in `projection/main.py`. `cli.py` presents results
and progress controls; `schema.py` defines method identities; `plans.py` writes
stage plans and coverage; `environment.py` records installed runtime versions.

[Browse the model × method matrix](../README.md).

`training/` supplies streamed data, disjoint splits, minibatch optimization,
validated eager inference, metric totals, and saved reports. Each model's
`main.py` explicitly shows export, calibration mappings, and Qraft recipes.
