"""Reproducible file selection and bounded image loading for Imagenette."""

from collections.abc import Callable, Iterable, Mapping
from hashlib import sha256
from heapq import nsmallest
from pathlib import Path

from PIL import Image
from torchvision.datasets import Imagenette

from examples._shared.vision.schema import ImageRecord
from qraft.domain import FloatArray

# The ten dataset labels must map back into the model's full ImageNet output space.
IMAGENETTE_CLASSES: Mapping[str, str] = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def download_dataset(cache: Path) -> Path:
    """Download the public archive once; let TorchVision verify its checksum."""
    root = cache / "imagenette"
    root.mkdir(parents=True, exist_ok=True)
    Imagenette(root=root, split="train", size="160px", download=True)
    return root / "imagenette2-160"


def iter_records(root: Path, categories: tuple[str, ...]) -> Iterable[ImageRecord]:
    """Read only file metadata; map WordNet folders to ImageNet class indices."""
    for wnid, name in IMAGENETTE_CLASSES.items():
        if name not in categories:
            raise ValueError(f"pretrained categories omit {name}")
        directory = root / wnid
        if not directory.is_dir():
            raise ValueError(f"missing dataset class directory: {directory}")
        for path in directory.iterdir():
            if path.suffix.lower() in (".jpeg", ".jpg"):
                yield ImageRecord(
                    path=path, label=categories.index(name), class_name=name
                )


def select_records(
    root: Path,
    categories: tuple[str, ...],
    count: int,
    seed: int,
) -> tuple[ImageRecord, ...]:
    """Select a seeded subset with O(count) metadata, ignoring listing order."""

    def rank(record: ImageRecord) -> bytes:
        """Hash relative identity so selections survive cache relocation."""
        identity = f"{seed}:{record.path.relative_to(root).as_posix()}"
        return sha256(identity.encode()).digest()

    selected = tuple(nsmallest(count, iter_records(root, categories), key=rank))
    if len(selected) != count:
        raise ValueError(
            f"requested {count} images but found {len(selected)} in {root}"
        )
    return selected


def read_input(
    record: ImageRecord, preprocess: Callable[[Image.Image], FloatArray]
) -> FloatArray:
    """Decode exactly one image and close its file before returning model input."""
    with Image.open(record.path) as image:
        return preprocess(image.convert("RGB"))


def calibration_source(
    records: tuple[ImageRecord, ...],
    preprocess: Callable[[Image.Image], FloatArray],
) -> Callable[[], Iterable[Mapping[str, FloatArray]]]:
    """Return a replay factory without retaining decoded images or activations."""

    def replay() -> Iterable[Mapping[str, FloatArray]]:
        """Re-read the same selected files using deterministic weight preprocessing."""
        for record in records:
            yield {"images": read_input(record, preprocess)}

    return replay
