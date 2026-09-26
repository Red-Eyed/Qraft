# Calibration

`collect` deduplicates `Requirement(tensor=..., axes=...)` requests and streams
samples through an injected `Evaluator`. Empty axes mean per-tensor reduction;
retained axes request channelwise extrema. Changing retained dimensions is an error.
Collection, replay, and calibration policy intervals return typed `Result` values;
empty sources, missing observations, invalid ranges, and out-of-range replay produce
`returns.result.Failure` with `QraftError` diagnostics. Evaluator plugins return
errors through the same contract. Statistics constructors retain Pydantic
validation exceptions.

`histograms` replays a deterministic source into fixed bins established from
first-pass extrema. Its local accumulators are mutable; returned statistics are
validated and read-only. Percentile calibration clips using enclosing bin edges.
MinMax does not request a histogram or a replay pass.

`Samples` is a factory, never a materialized dataset. Collection retains only
current activations and aggregate statistics. Histogram storage scales with bins
and requested tensors, not dataset length. Runtime execution is injected; the
numerical reducers and clipping policies know nothing about ONNX or files.
