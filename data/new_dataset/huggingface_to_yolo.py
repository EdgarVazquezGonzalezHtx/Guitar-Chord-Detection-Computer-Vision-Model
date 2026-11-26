#!/usr/bin/env python3
"""Downloads a Hugging Face image-classification dataset and write it in YOLO layout.

Each image gets a whole-image box ("<class> 0.5 0.5 1.0 1.0"), because the
Hugging Face dataset has one chord label per image and no boxes. The output
can then be merged with data/new_dataset/mergehf_with_orig.py.

Example (run from the project root):
    python data/new_dataset/huggingface_to_yolo.py dduka/guitar-chords --out data/extra1
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

SPLIT_MAP = {"train": "train", "validation": "valid", "valid": "valid", "val": "valid", "test": "test"}
SPLITS = ("train", "valid", "test")

def convert(ds, out: Path, image_column: str = "image", label_column: str = "label") -> list[str]:
    """Write a datasets.DatasetDict to `out` in YOLO layout and return the class names."""
    first = next(iter(ds.values()))
    names = list(first.features[label_column].names)
    print(f"Classes ({len(names)}): {names}")

    for split in SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    for split, data in ds.items():
        dest = SPLIT_MAP.get(split)
        if dest is None:
            print(f"  {split}: unknown split name, skipped")
            continue
        counts = Counter()
        for i, row in enumerate(data):
            stem = f"{split}_{i:05d}"
            label = int(row[label_column])
            row[image_column].convert("RGB").save(out / dest / "images" / f"{stem}.jpg", quality=95)
            (out / dest / "labels" / f"{stem}.txt").write_text(f"{label} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
            counts[names[label]] += 1
        arrow = f" -> {dest}" if dest != split else ""
        print(f"  {split}{arrow}: {sum(counts.values())} images  {dict(sorted(counts.items()))}")

    data_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(names),
        "names": names,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return names

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="Hugging Face dataset id, e.g. dduka/guitar-chords")
    parser.add_argument("--out", required=True, type=Path, help="new folder to write the YOLO dataset to")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    if args.out.exists():
        sys.exit(f"{args.out} already exists; delete it or choose another --out folder")

    from datasets import load_dataset  # imported here so --help works without the package

    ds = load_dataset(args.repo)
    convert(ds, args.out, args.image_column, args.label_column)
    print(f"\nWrote {args.out}. Next: python data/new_dataset/mergehf_with_orig.py --base data/raw --add {args.out} "
          f"--out data/merged --keep-base-test")

if __name__ == "__main__":
    main()
