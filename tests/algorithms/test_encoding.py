"""Affine quantization answers checked without graph execution."""

import numpy as np
import pytest

from quantsmith.algorithms.encoding import encode
from quantsmith.calibration import MinMaxStats
from quantsmith.domain import IntegerType, PerChannel, PerTensor


@pytest.mark.parametrize(
    ("dtype", "symmetric", "low", "high", "scale", "zero"),
    [
        (IntegerType.INT8, True, -127.0, 127.0, 1.0, 0),
        (IntegerType.UINT8, True, -127.0, 127.0, 1.0, 128),
        (IntegerType.INT8, False, -128.0, 127.0, 1.0, 0),
        (IntegerType.UINT8, False, -2.0, 3.0, 5 / 255, 102),
        (IntegerType.UINT8, False, 1.0, 3.0, 3 / 255, 0),
        (IntegerType.UINT8, False, -3.0, -1.0, 3 / 255, 255),
    ],
    ids=[
        "signed-symmetric",
        "unsigned-symmetric",
        "signed-asymmetric",
        "unsigned-asymmetric",
        "positive-includes-zero",
        "negative-includes-zero",
    ],
)
def test_known_affine_parameters(
    dtype: IntegerType,
    symmetric: bool,
    low: float,
    high: float,
    scale: float,
    zero: int,
) -> None:
    """Check independently calculated scales and zero points, including real zero."""
    stats = MinMaxStats(
        minimum=np.asarray(low, dtype=np.float32),
        maximum=np.asarray(high, dtype=np.float32),
    )
    encoding = encode(stats, dtype, symmetric, PerTensor())
    assert float(encoding.scale) == pytest.approx(scale)
    assert int(encoding.zero_point) == zero
    assert encoding.zero_point.dtype == np.dtype(dtype.value)
    qmin, qmax = dtype.bounds
    assert qmin <= zero <= qmax


@pytest.mark.parametrize(
    ("dtype", "symmetric", "zero"),
    [
        (IntegerType.INT8, False, -128),
        (IntegerType.UINT8, False, 0),
        (IntegerType.INT8, True, 0),
        (IntegerType.UINT8, True, 128),
    ],
)
def test_all_zero_range(dtype: IntegerType, symmetric: bool, zero: int) -> None:
    """Zero data gets a unit scale and exactly representable zero."""
    stats = MinMaxStats(
        minimum=np.asarray(0, dtype=np.float32), maximum=np.asarray(0, dtype=np.float32)
    )
    encoding = encode(stats, dtype, symmetric, PerTensor())
    assert float(encoding.scale) == 1
    assert int(encoding.zero_point) == zero


def test_per_channel_parameters_and_ownership() -> None:
    """Use independent channel scales without aliasing statistics or output storage."""
    low = np.asarray([-127, 0, -2], dtype=np.float32)
    high = np.asarray([127, 0, 3], dtype=np.float32)
    stats = MinMaxStats(minimum=low, maximum=high)
    encoding = encode(stats, IntegerType.INT8, True, PerChannel(axis=1))
    np.testing.assert_allclose(encoding.scale, [1, 1, 3 / 127])
    np.testing.assert_array_equal(encoding.zero_point, [0, 0, 0])
    np.testing.assert_array_equal(low, [-127, 0, -2])
    np.testing.assert_array_equal(high, [127, 0, 3])
    assert not np.shares_memory(encoding.scale, stats.maximum)
    assert not encoding.scale.flags.writeable
    assert not encoding.zero_point.flags.writeable
