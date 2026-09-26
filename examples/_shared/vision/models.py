"""TorchVision model construction and validated PyTorch/ONNX boundaries."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import Tensor, nn

from qraft.domain import FloatArray


class LoadedModel:
    """Hold a concrete Torch model and its matching pretrained preprocessing."""

    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[Image.Image], object],
        categories: tuple[str, ...],
        weights: str,
    ) -> None:
        """Bind the exact weight recipe to the model used for export."""
        self.model = model.eval().cpu()
        self.transform = transform
        self.categories = categories
        self.weights = weights

    def preprocess(self, image: Image.Image) -> FloatArray:
        """Validate transformed data before handing float32 NCHW inputs to ORT."""
        value: object = self.transform(image)
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise ValueError("preprocessing must return a CHW tensor")
        array = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)[None, ...]
        if array.shape[1] != 3 or not np.all(np.isfinite(array)):
            raise ValueError("preprocessing returned invalid RGB data")
        return np.ascontiguousarray(array)


def checked_categories(value: object) -> tuple[str, ...]:
    """Validate the untyped TorchVision metadata at its source boundary."""
    if not isinstance(value, list) or len(value) != 1000:
        raise ValueError("expected 1000 ImageNet categories")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError("invalid ImageNet category name")
        result.append(item)
    return tuple(result)


def checked_logits(value: object) -> FloatArray:
    """Admit only finite batch-one ImageNet logits from an external runtime."""
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise ValueError("expected float32 NumPy logits")
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (1, 1000) or not np.all(np.isfinite(array)):
        raise ValueError("expected finite logits with shape (1, 1000)")
    return array


class TorchRunner:
    """Execute a real pretrained module without gradients or graph compilation."""

    def __init__(self, model: nn.Module) -> None:
        """Use the already loaded CPU evaluation model."""
        self.model = model

    def __call__(self, inputs: FloatArray) -> FloatArray:
        """Validate model outputs after eager inference."""
        with torch.inference_mode():
            output: object = self.model(torch.from_numpy(inputs))
        if not isinstance(output, Tensor):
            raise ValueError("expected a single Torch output tensor")
        return checked_logits(output.detach().cpu().numpy())


class OrtRunner:
    """Run a deployment-style optimized CPU session for final model evaluation."""

    def __init__(self, path: Path, threads: int) -> None:
        """Create a fixed-thread session before any latency measurements."""
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

    def __call__(self, inputs: FloatArray) -> FloatArray:
        """Validate the dynamic ORT response at the adapter boundary."""
        outputs: object = self.session.run(["logits"], {"images": inputs})
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise ValueError("expected one ONNX output")
        return checked_logits(outputs[0])
