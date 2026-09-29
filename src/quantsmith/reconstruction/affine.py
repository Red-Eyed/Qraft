"""Native-array affine quantization shared by reconstruction methods."""

import numpy as np
from numpy.typing import NDArray

from quantsmith.domain import Encoding, FloatArray, IntArray, PerChannel, PerTensor


def parameters(encoding: Encoding, rank: int) -> tuple[FloatArray, IntArray]:
    """Bind channel parameters to their broadcast shape."""
    match encoding.granularity:
        case PerTensor():
            return encoding.scale, encoding.zero_point
        case PerChannel(axis=axis):
            shape = [1] * rank
            shape[axis] = encoding.scale.size
            return encoding.scale.reshape(shape), encoding.zero_point.reshape(shape)


def codes(value: FloatArray | NDArray[np.float64], encoding: Encoding) -> IntArray:
    """Round to nearest even on the exact export grid and saturate to storage."""
    scale, zero = parameters(encoding, value.ndim)
    bounds = np.iinfo(encoding.zero_point.dtype)
    integers = np.clip(np.rint(value / scale) + zero, bounds.min, bounds.max)
    if encoding.zero_point.dtype == np.int8:
        return integers.astype(np.int8)
    return integers.astype(np.uint8)


def restore(value: IntArray, encoding: Encoding) -> FloatArray:
    """Dequantize integer codes using FP32 arithmetic like the exported graph."""
    scale, zero = parameters(encoding, value.ndim)
    return (value.astype(np.float32) - zero.astype(np.float32)) * scale


def quantize(value: FloatArray, encoding: Encoding) -> FloatArray:
    """Simulate the final affine quantizer without changing caller storage."""
    return restore(codes(value, encoding), encoding)
