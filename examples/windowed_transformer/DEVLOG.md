# Windowed causal Transformer development log

## 2026-09-26 — Readable standalone demonstration

The local model and entry point replace the previous shared suite wrapper. Model
construction, named ONNX inputs, calibration replay, and all three method recipes
are visible together so readers can adapt them. Architecture dimensions and
training/data split policies are retained; parameter paths now belong to the local
model definition. Existing checkpoints from the older suite may use different keys.

Offline checks cover ONNX/INT8 numerical round trips and, for Transformers, causal
behavior and float dynamic-attention edges. Small smoke runs verify the complete
CLI/report path; they do not establish production accuracy or deployment speed.
