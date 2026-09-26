# Plans

A `QuantizationPlan` is an inspectable set of decisions before backend mutation.
`QuantizeInput` owns one consumer's encoding. `RescaleInput` describes a paired
activation/weight transformation. These are closed alternatives so new operation
kinds expose incomplete lowering consumers to static checking.

`then` combines compatible patches and rejects duplicate writes, exclusions with
operations, and mixed transform/quantization stages. The application lowers a
transform stage before recalibrating the next revision. Plans do not contain I/O
or backend objects, and must be applied to their original graph revision.

`then` returns `qraft.result.Ok(plan)` or `Err(QraftError(...))` with
the `conflict` category. Direct plan constructors use normal Pydantic validation
exceptions.
