"""Streaming classification metrics and warmed-up batch-one measurements."""

from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from rich.progress import Progress

from examples._shared.schema import Latency, Method
from examples._shared.vision.data import read_input
from examples._shared.vision.schema import (
    DemoConfig,
    GalleryItem,
    ImageRecord,
    Metrics,
    Prediction,
)
from qraft.domain import FloatArray

Runner = Callable[[FloatArray], FloatArray]


class ErrorTotals(BaseModel):
    """Accumulate elementwise error without retaining prediction arrays."""

    squared: float = Field(default=0)
    maximum: float = Field(default=0)
    elements: int = Field(default=0)
    agreement: int = Field(default=0)

    def update(self, actual: FloatArray, reference: FloatArray) -> None:
        """Aggregate in float64 to avoid float32 accumulation drift."""
        if actual.shape != reference.shape:
            raise ValueError("prediction shapes differ")
        difference = actual.astype(np.float64) - reference
        self.squared += float(np.sum(difference * difference))
        self.maximum = max(self.maximum, float(np.max(np.abs(difference))))
        self.elements += difference.size
        self.agreement += int(np.argmax(actual) == np.argmax(reference))


class MetricTotals(BaseModel):
    """Track classification and two distinct reference comparisons."""

    samples: int = Field(default=0)
    top1: int = Field(default=0)
    top5: int = Field(default=0)
    torch: ErrorTotals = Field(default_factory=ErrorTotals)
    onnx: ErrorTotals = Field(default_factory=ErrorTotals)

    def update(
        self, actual: FloatArray, torch: FloatArray, onnx: FloatArray, label: int
    ) -> None:
        """Evaluate all 1000 classes rather than restricting to Imagenette's ten."""
        self.samples += 1
        self.top1 += int(int(np.argmax(actual)) == label)
        self.top5 += int(label in np.argsort(actual[0])[-5:])
        self.torch.update(actual, torch)
        self.onnx.update(actual, onnx)

    def finish(self) -> Metrics:
        """Return a validated report only when observations exist."""
        if not self.samples:
            raise ValueError("cannot report empty evaluation")
        return Metrics(
            samples=self.samples,
            top1_accuracy=self.top1 / self.samples,
            top5_accuracy=self.top5 / self.samples,
            agreement_with_torch=self.torch.agreement / self.samples,
            agreement_with_onnx=self.onnx.agreement / self.samples,
            mse_vs_torch=self.torch.squared / self.torch.elements,
            mse_vs_onnx=self.onnx.squared / self.onnx.elements,
            max_abs_vs_torch=self.torch.maximum,
            max_abs_vs_onnx=self.onnx.maximum,
        )


def benchmark(runner: Runner, inputs: FloatArray, warmup: int, runs: int) -> Latency:
    """Time warmed inference, excluding image loading and session startup."""
    for _ in range(warmup):
        runner(inputs)
    timings = np.empty(runs, dtype=np.float64)
    for index in range(runs):
        start = perf_counter()
        runner(inputs)
        timings[index] = (perf_counter() - start) * 1000
    return Latency(
        median_ms=float(np.median(timings)),
        p95_ms=float(np.percentile(timings, 95)),
        runs=runs,
    )


def prediction(
    method: Method, logits: FloatArray, label: int, categories: tuple[str, ...]
) -> Prediction:
    """Summarize one image without implying that softmax is calibrated confidence."""
    values = logits[0].astype(np.float64)
    scores = np.exp(values - np.max(values))
    scores /= scores.sum()
    chosen = int(np.argmax(values))
    return Prediction(
        method=method,
        label=chosen,
        name=categories[chosen],
        confidence=float(scores[chosen]),
        correct=chosen == label,
    )


def save_gallery_image(record: ImageRecord, destination: Path) -> None:
    """Save a bounded preview and close the original dataset file immediately."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(record.path) as source:
        image = source.convert("RGB")
        image.thumbnail((320, 240))
        image.save(destination, quality=85)


def measure(
    runners: Mapping[Method, Runner],
    records: tuple[ImageRecord, ...],
    preprocess: Callable[[Image.Image], FloatArray],
    categories: tuple[str, ...],
    model_dir: Path,
    config: DemoConfig,
    progress: Progress,
) -> tuple[dict[Method, Metrics], tuple[GalleryItem, ...]]:
    """Evaluate identical held-out images across all variants in a single traversal."""
    totals = {method: MetricTotals() for method in runners}
    gallery: list[GalleryItem] = []
    task = progress.add_task("Held-out evaluation: all variants", total=len(records))
    for index, record in enumerate(records):
        inputs = read_input(record, preprocess)
        outputs = {method: runner(inputs) for method, runner in runners.items()}
        for method, output in outputs.items():
            totals[method].update(
                output, outputs[Method.TORCH], outputs[Method.ONNX], record.label
            )
        if index < config.gallery_images:
            image_path = model_dir / "images" / f"{index:03d}.jpg"
            save_gallery_image(record, image_path)
            gallery.append(
                GalleryItem(
                    image=image_path.relative_to(config.output),
                    truth=record.class_name,
                    predictions=tuple(
                        prediction(method, output, record.label, categories)
                        for method, output in outputs.items()
                    ),
                )
            )
        progress.advance(task)
    progress.remove_task(task)
    return {method: total.finish() for method, total in totals.items()}, tuple(gallery)
