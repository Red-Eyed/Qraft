"""TorchVision model construction and validated PyTorch/ONNX boundaries."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from PIL import Image
from pydantic import ConfigDict, Field, TypeAdapter
from torch import Tensor, nn
from torchvision.models import WeightsEnum

from qraft.domain import FloatArray

_TENSOR_OUTPUT = TypeAdapter(
    Tensor, config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)
_ARRAY_OUTPUT = TypeAdapter(
    NDArray[np.generic], config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)
_LOGIT_OUTPUTS = TypeAdapter[list[NDArray[np.generic]]](
    Annotated[list[NDArray[np.generic]], Field(min_length=1, max_length=1)],
    config=ConfigDict(strict=True, arbitrary_types_allowed=True),
)
_CATEGORIES = TypeAdapter[list[str]](
    Annotated[
        list[Annotated[str, Field(min_length=1)]],
        Field(min_length=1000, max_length=1000),
    ],
    config=ConfigDict(strict=True),
)


class TorchImageTransform:
    """Validate TorchVision's dynamic output behind a tensor-returning callable."""

    def __init__(self, transform: nn.Module) -> None:
        """Retain the checkpoint's exact preprocessing module."""
        self.transform = transform

    def __call__(self, image: Image.Image) -> Tensor:
        """Reject non-tensor output before it reaches application preprocessing."""
        return _TENSOR_OUTPUT.validate_python(self.transform(image))

    def __repr__(self) -> str:
        """Keep the checkpoint recipe visible in saved example reports."""
        return repr(self.transform)


class LoadedModel:
    """Hold a concrete Torch model and its matching pretrained preprocessing."""

    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[Image.Image], Tensor],
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
        value = self.transform(image)
        if value.ndim != 3:
            raise ValueError("preprocessing must return a CHW tensor")
        array = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)[None, ...]
        if array.shape[1] != 3 or not np.all(np.isfinite(array)):
            raise ValueError("preprocessing returned invalid RGB data")
        return np.ascontiguousarray(array)


def checked_categories(weights: WeightsEnum) -> tuple[str, ...]:
    """Validate the untyped TorchVision metadata at its source boundary."""
    return tuple(_CATEGORIES.validate_python(weights.meta["categories"]))


def _checked_logits(value: NDArray[np.generic]) -> FloatArray:
    """Admit only finite batch-one ImageNet logits from an external runtime."""
    if value.dtype != np.float32:
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
            output = _TENSOR_OUTPUT.validate_python(
                self.model(torch.from_numpy(inputs))
            )
        return _checked_logits(
            _ARRAY_OUTPUT.validate_python(output.detach().cpu().numpy())
        )


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
        outputs = _LOGIT_OUTPUTS.validate_python(
            self.session.run(["logits"], {"images": inputs})
        )
        return _checked_logits(outputs[0])
