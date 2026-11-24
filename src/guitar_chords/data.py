"""Load chord datasets with supplied splits or class folders."""

from pathlib import Path
import random

from . import CHORDS

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_yolo_splits(root: Path):
    """Read the export's train, validation and test samples in that order."""
    import yaml

    metadata_path = root / "data.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    names = metadata.get("names") if isinstance(metadata, dict) else None
    if (not isinstance(names, list) or len(names) != len(CHORDS)
            or any(not isinstance(name, str) for name in names)
            or set(names) != set(CHORDS)):
        raise ValueError(f"{metadata_path}: expected the 11 chord names")
    class_indices = [CHORDS.index(name) for name in names]

    # The export's YAML paths are relative to its original layout.
    return tuple(_read_yolo_split(root / split, class_indices)
                 for split in ("train", "valid", "test"))


def _read_yolo_split(directory, class_indices):
    images = sorted(path for path in (directory / "images").glob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"{directory / 'images'}: no supported images found")

    samples = []
    for image in images:
        label_path = directory / "labels" / f"{image.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"{image}: missing label file {label_path}")
        rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        try:
            class_ids = {int(row[0]) for row in rows}
        except ValueError as exc:
            raise ValueError(f"{label_path}: class IDs must be integers") from exc
        if len(class_ids) != 1:
            raise ValueError(f"{label_path}: expected one distinct chord class")
        class_id = class_ids.pop()
        if not 0 <= class_id < len(class_indices):
            raise ValueError(f"{label_path}: chord class ID {class_id} is out of range")
        samples.append((str(image), class_indices[class_id]))
    return samples


def resnet18_preprocessing():
    """Build the deterministic input transform for pretrained ResNet-18."""
    from torchvision.models import ResNet18_Weights

    return ResNet18_Weights.DEFAULT.transforms()


class ChordImageDataset:
    """Map-style PyTorch dataset returning an RGB tensor and chord index."""

    def __init__(self, samples):
        self.samples = samples
        self.transform = resnet18_preprocessing()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image, ImageOps

        path, label = self.samples[index]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image), label


def split_dataset(root: Path, seed: int = 42):
    """Return train/validation/test (path, class-index) lists; never move files."""
    splits = ([], [], [])
    rng = random.Random(seed)
    for label, chord in enumerate(CHORDS):
        images = sorted(p for p in (root / chord).glob("*")
                        if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) < 10:
            raise ValueError(f"{root / chord}: need at least 10 images per chord; "
                             "the dataset is a placeholder until you add it.")
        rng.shuffle(images)
        train_end = int(len(images) * .7)
        val_end = train_end + int(len(images) * .2)
        for target, group in zip(splits, (images[:train_end], images[train_end:val_end], images[val_end:])):
            target.extend((str(path), label) for path in group)
    return splits
