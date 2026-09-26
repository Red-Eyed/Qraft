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

## NumPy, Torch, and GPU calibration

Calibration currently uses NumPy for extrema, histograms, and scale calculations.
This fits the CPU array interface and keeps Torch optional. The ONNX evaluator
executes the model through ONNX Runtime on CPU with one intra-op thread; NumPy
does not execute the model. Replacing NumPy with CPU Torch tensors alone does not
guarantee a speedup.

For large-scale GPU calibration, the important constraint is where activations
live and how often they move. An FP32 activation shaped `[8, 2048, 4096]` occupies
256 MiB. Computing minima and maxima per channel on the GPU reduces that to
32 KiB of statistics, avoiding a full activation transfer to CPU for each batch.
Small scale and zero-point arrays are reasonable NumPy workloads; repeatedly
moving full activations to CPU is the concern.

The intended direction is device-resident execution and statistics reduction:
use Torch tensors for GPU extrema and histogram accumulation, with ONNX Runtime
GPU execution and I/O binding where supported. NumPy can remain at serialization
boundaries. [ONNX Runtime supports binding GPU Torch tensors directly](https://onnxruntime.ai/docs/api/python/api_summary).
This requires extending the current NumPy-based evaluator and statistics
contracts, not just changing imports or selecting a GPU execution provider.
Keep sample processing streamed and aggregate storage bounded on the device.

This GPU path is not implemented or benchmarked. Measure execution, reductions,
transfers, and peak memory separately before claiming a speedup; the current
instrumented evaluator is not a deployment-performance benchmark.
