"""Pure affine quantization parameter calculations."""

import numpy as np

from qraft.calibration import MinMaxStats
from qraft.domain import Encoding, Granularity, IntegerType


def encode(
    stats: MinMaxStats, dtype: IntegerType, symmetric: bool, granularity: Granularity
) -> Encoding:
    """Include real zero; use unit scales for all-zero channels."""
    low = np.minimum(stats.minimum.astype(np.float64), 0)
    high = np.maximum(stats.maximum.astype(np.float64), 0)
    qmin, qmax = dtype.bounds
    if symmetric:
        bound = np.maximum(np.abs(low), np.abs(high))
        midpoint = 0 if dtype is IntegerType.INT8 else 128
        denominator = min(qmax - midpoint, midpoint - qmin)
        scale = bound / denominator
        zero = np.full_like(scale, midpoint)
    else:
        scale = (high - low) / (qmax - qmin)
        scale = np.where(scale == 0, 1, scale)
        zero = np.clip(np.rint(qmin - low / scale), qmin, qmax)
    scale = np.where(scale == 0, 1, scale).astype(np.float32)
    return Encoding(
        scale=scale,
        zero_point=np.asarray(zero, dtype=np.int8)
        if dtype is IntegerType.INT8
        else np.asarray(zero, dtype=np.uint8),
        granularity=granularity,
    )
