# Domain

Pure graph records, integer types, granularity variants, and affine encodings.
There is no ONNX, runtime, filesystem, or application-path dependency here.

Construct models with keyword arguments. Strict frozen Pydantic models and `Field`
declarations enforce record shapes. Array validators additionally check dtype,
finiteness, positivity, shape, and ownership where appropriate. Graph constants
and encodings are detached and read-only. These runtime checks complement static
annotations; ndarray annotations alone do not prove a shape or dtype.

Graph constants preserve valid IEEE infinities, such as additive causal masks.
Quantization algorithms enforce finite values when collecting selected weights
and activations; statistics and encodings also require finite values. Int64 graph
inputs support embedding-based models without changing floating-point statistics.
