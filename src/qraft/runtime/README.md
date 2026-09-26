# Runtime

`OnnxEvaluator` implements the calibration evaluator contract using CPU ORT.
It owns its graph, instruments only requested FP32 outputs, validates returned
arrays, and caches at most one session. `run` returns `Result` with documented ORT
execution failures and tensor validation failures. Graph construction validates
its input and can raise. Samples may use FP32 features or int64 token IDs; observed
outputs are FP32. Samples are neither mutated nor retained.

`evaluate` depends only on the evaluator protocol and streams reference/candidate
outputs into aggregate MSE and maximum absolute error. It rejects shape mismatches
and empty evaluation with `qraft.result.Err` carrying a `QraftError`.
Candidate time includes session construction; optimizations
are disabled for activation observability, so this is not a deployment benchmark.
