"""Small deterministic reconstruction inputs with nontrivial channel correlation."""

import numpy as np
import pytest

from qraft.domain import Encoding, PerChannel, PerTensor
from qraft.reconstruction import Batch, Linear, Problem, Replay
from qraft.result import Ok


@pytest.fixture
def problem() -> Problem:
    """Use deliberately coarse grids so rounding differences remain measurable."""
    return Problem(
        weight=np.array([[0.49, 0.38, -0.8], [0.71, -0.49, 0.21]], dtype=np.float32),
        weight_encoding=Encoding(
            scale=np.array([0.2, 0.2], dtype=np.float32),
            zero_point=np.zeros(2, dtype=np.int8),
            granularity=PerChannel(axis=0),
        ),
        activation_encoding=Encoding(
            scale=np.array(0.02, dtype=np.float32),
            zero_point=np.array(0, dtype=np.int8),
            granularity=PerTensor(),
        ),
        operator=Linear(),
    )


@pytest.fixture
def batch() -> Batch:
    """Introduce correlated upstream error independently of local quantization."""
    rng = np.random.default_rng(12)
    reference = (rng.normal(size=(32, 3)) * 0.4).astype(np.float32)
    candidate = reference @ np.array(
        [[0.9, 0.2, 0], [0, 1.1, 0.1], [0.1, 0, 0.8]], dtype=np.float32
    )
    return Batch(reference=reference, candidate=candidate)


@pytest.fixture
def asymmetric_batch() -> Batch:
    """Make upstream error in channel zero recoverable through channel one."""
    candidate = np.eye(3, dtype=np.float32)
    reference = candidate.copy()
    reference[:, 0] += candidate[:, 1] * np.float32(0.5)
    return Batch(reference=reference, candidate=candidate)


@pytest.fixture
def replay(batch: Batch) -> Replay:
    """Provide a reusable numerical batch without runtime or filesystem services."""
    return lambda: (Ok(batch),)
