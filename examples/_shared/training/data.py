"""Real data admission with disjoint splits and training-only preprocessing."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter
from returns.result import Failure, Result, Success
from sklearn.datasets import load_wine

from examples._shared.training.schema import Config, Dataset, Split, TaskKind, Tokens
from qraft.domain import FloatArray
from qraft.result import FailureKind, QraftError, failure, validate

TEXT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt"
)
TEXT_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

_WINE_ARRAYS = TypeAdapter(
    tuple[NDArray[np.generic], NDArray[np.generic]],
    config=ConfigDict(strict=True, arbitrary_types_allowed=True),
)


def wine_arrays() -> tuple[FloatArray, Tokens]:
    """Validate sklearn's external tuple before exposing native typed arrays."""
    features, labels = _WINE_ARRAYS.validate_python(load_wine(return_X_y=True))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def wine(config: Config) -> Dataset:
    """Stratify the finite 178-row dataset; fit normalization on training rows only."""
    features, targets = wine_arrays()
    rng = np.random.default_rng(config.seed)
    train: list[int] = []
    calibration: list[int] = []
    evaluation: list[int] = []
    for label in range(3):
        indices = rng.permutation(np.flatnonzero(targets == label))
        first, second = int(len(indices) * 0.6), int(len(indices) * 0.8)
        train.extend(int(i) for i in indices[:first])
        calibration.extend(int(i) for i in indices[first:second])
        evaluation.extend(int(i) for i in indices[second:])
    mean, std = features[train].mean(axis=0), features[train].std(axis=0)
    normalized = ((features - mean) / np.maximum(std, 1e-6)).astype(np.float32)

    def split(indices: list[int]) -> Split:
        """Bind rows and labels to their original dataset identities."""
        return Split(
            inputs=normalized[indices],
            targets=targets[indices],
            identities=tuple(indices),
        )

    return Dataset(
        kind=TaskKind.WINE,
        train=split(train),
        calibration=split(calibration),
        evaluation=split(evaluation),
        labels=("cultivar 0", "cultivar 1", "cultivar 2"),
        source="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html",
        sha256=hashlib.sha256(features.tobytes() + targets.tobytes()).hexdigest(),
        preprocessing=f"train-only mean={mean.tolist()}, std={std.tolist()}",
        train_end=len(train),
        calibration_end=len(train) + len(calibration),
    )


def cached_text(cache: Path) -> str:
    """Download at most 2 MB from a pinned revision and verify cached content."""
    destination = cache / "tinyshakespeare.txt"
    cache.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urlopen(TEXT_URL, timeout=60) as response:
            payload = response.read(2_000_001)
        if hashlib.sha256(payload).hexdigest() != TEXT_SHA:
            raise ValueError("Shakespeare download failed its checksum")
        destination.write_bytes(payload)
    payload = destination.read_bytes()
    if hashlib.sha256(payload).hexdigest() != TEXT_SHA:
        raise ValueError("cached Shakespeare text failed its checksum")
    return payload.decode("utf-8")


def text_windows(
    tokens: Tokens, start: int, end: int, count: int, length: int
) -> Result[Split, QraftError]:
    """Select nonoverlapping windows entirely inside one contiguous data split."""
    available = (end - start - 1) // length
    if count > available:
        return failure(
            FailureKind.INVALID_DATA,
            "text sampling",
            f"requested {count} windows, only {available} available",
        )
    offsets = np.linspace(start, end - length - 1, count, dtype=np.int64)
    return Success(
        Split(
            inputs=np.stack([tokens[i : i + length] for i in offsets]),
            targets=np.stack([tokens[i + 1 : i + length + 1] for i in offsets]),
            identities=tuple(int(i) for i in offsets),
        )
    )


def shakespeare(
    config: Config, kind: TaskKind = TaskKind.TEXT
) -> Result[Dataset, QraftError]:
    """Fit vocabulary on training text; never train on calibration/evaluation spans."""
    match validate("Shakespeare data", lambda: cached_text(config.cache)):
        case Failure() as error:
            return error
        case _ as resolved:
            text = resolved.unwrap()
    first, second = int(len(text) * 0.8), int(len(text) * 0.9)
    labels = tuple(sorted(set(text[:first])))
    vocabulary = {char: i for i, char in enumerate(labels)}
    if set(text) - vocabulary.keys():
        return failure(
            FailureKind.INVALID_DATA,
            "vocabulary",
            "held-out text contains unseen characters",
        )
    tokens = np.asarray([vocabulary[char] for char in text], dtype=np.int64)
    match text_windows(
        tokens, first, second, config.calibration_samples, config.sequence_length
    ):
        case Failure() as error:
            return error
        case _ as resolved:
            calibration = resolved.unwrap()
    match text_windows(
        tokens, second, len(tokens), config.evaluation_samples, config.sequence_length
    ):
        case Failure() as error:
            return error
        case _ as resolved:
            evaluation = resolved.unwrap()
    return Success(
        Dataset(
            kind=kind,
            train=Split(
                inputs=tokens[: first - 1], targets=tokens[1:first], identities=()
            ),
            calibration=calibration,
            evaluation=evaluation,
            labels=labels,
            source=TEXT_URL,
            sha256=TEXT_SHA,
            preprocessing=(
                "train-only character vocabulary; int64 token IDs; "
                "independent fixed-context windows"
            ),
            train_end=first,
            calibration_end=second,
        )
    )
