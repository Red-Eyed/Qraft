# Pretrained model experiment log

## 2026-09-26 — Real CNN exports and held-out measurements

Added explicit TorchVision ImageNet checkpoints for ResNet-18, MobileNetV2, and
EfficientNet-B0. Every model uses its matching preprocessing and all 1,000 classes.
Imagenette train images calibrate; val images evaluate. Seeded selection stores
only bounded file metadata, and inference streams one decoded image at a time.

The full Qraft 0.1.1 run used 128 calibration and 256 held-out images. Torch/FP32
ONNX top-1 accuracy was 66.02%, 64.06%, and 75.00% respectively; export MSE was below
1.4e-11. MinMax top-1 was 67.58%, 63.28%, and 44.92%. Percentile gave 67.19%, 62.50%,
and 69.53%. SmoothQuant gave 66.02%, 65.23%, and 50.00%. The EfficientNet regressions
remain visible in reports; no subset or gallery was chosen by prediction outcome.

One-thread optimized CPU ORT measurements showed different kernel behavior across
architectures: MobileNetV2 improved from about 4.41 ms FP32 to 3.37 ms for ordinary
QDQ; ResNet-18 and EfficientNet-B0 slowed down. SmoothQuant does not universally
improve accuracy or speed. QDQ retains original float weights; the example shell
prunes only dead initializer copies after private rescaling.

After the Qraft 0.2.0 Result migration, all three entry points were checked again
with four calibration and eight held-out images. Full 0.1.1 measurements retain
their original provenance; smoke output lives in a separate directory. Numerical
regressions additionally cover real offline Torch exports in the test suite.
