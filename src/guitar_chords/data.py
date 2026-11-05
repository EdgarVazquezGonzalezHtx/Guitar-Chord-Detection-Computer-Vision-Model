"""Discover class folders and create reproducible, stratified splits."""

from pathlib import Path
import random

from . import CHORDS


def split_dataset(root: Path, seed: int = 42):
    """Return train/validation/test (path, class-index) lists; never move files."""
    splits = ([], [], [])
    rng = random.Random(seed)
    for label, chord in enumerate(CHORDS):
        images = sorted(p for p in (root / chord).glob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(images) < 10:
            raise ValueError(f"{root / chord}: need at least 10 images per chord; "
                             "the dataset is a placeholder until you add it.")
        rng.shuffle(images)
        train_end = int(len(images) * .7)
        val_end = train_end + int(len(images) * .2)
        for target, group in zip(splits, (images[:train_end], images[train_end:val_end], images[val_end:])):
            target.extend((str(path), label) for path in group)
    return splits
