#!/usr/bin/env python3
"""Merges extra YOLO-format dataset into the base guitar chord dataset.
 
Class IDs in the added datasets are remapped by chord name to the base
dataset's class list, so "Am" in every dataset gets the same ID. Images whose
labels use chords not in the base list, have no label, or contain more than one
distinct chord are skipped, matching what the training code accepts.

Example (run from the project root):
    python data/new_dataset/mergehf_with_orig.py --base data/raw --add data/extra1 --out data/merged --keep-base-test
    guitar-chords train --data data/merged --epochs 20 --lr 1e-4 --unfreeze --checkpoint artifacts/merged_ft.pt
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")

def load_names(root: Path) -> list[str]:
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):  # some exports use {0: "A", 1: "Am", ...}
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]

def parse_aliases(pairs: list[str]) -> dict[str, str]:
    aliases = {}
    for pair in pairs:
        if "=" not in pair:
            sys.exit(f"--alias must look like OLD=NEW, got {pair!r}")
        old, new = pair.split("=", 1)
        aliases[old.strip()] = new.strip()
    return aliases

def convert_label(label: Path, remap: dict[int, int]) -> list[str] | None:
    """Return remapped label lines, or None if the image should be skipped."""
    lines = []
    for raw in label.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            return None
        if class_id not in remap:
            return None
        lines.append(" ".join([str(remap[class_id]), *parts[1:]]))
    if not lines or len({line.split()[0] for line in lines}) != 1:
        return None
    return lines

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="dataset whose chord list is kept (e.g. data/raw)")
    parser.add_argument("--add", required=True, type=Path, nargs="+", help="extra YOLO dataset folder(s) to merge in")
    parser.add_argument("--out", required=True, type=Path, help="new folder to write the merged dataset to")
    parser.add_argument("--alias", action="append", default=[], metavar="OLD=NEW",
                        help="rename a chord in the added datasets, e.g. --alias 'A minor=Am' (repeatable)")
    parser.add_argument("--keep-base-test", action="store_true",
                        help="put the added datasets' test images into valid, so the test set stays "
                             "identical to the base dataset and scores stay comparable with your baseline")
    args = parser.parse_args()

    if args.out.exists():
        sys.exit(f"{args.out} already exists; delete it or choose another --out folder")

    names = load_names(args.base)
    index = {name: i for i, name in enumerate(names)}
    aliases = parse_aliases(args.alias)

    sources = [("base", args.base)] + [(f"add{i}", root) for i, root in enumerate(args.add, 1)]
    for tag, root in sources:
        src_names = load_names(root)
        remap = {i: index[aliases.get(n, n)] for i, n in enumerate(src_names) if aliases.get(n, n) in index}
        unknown = [n for i, n in enumerate(src_names) if i not in remap]
        print(f"[{tag}] {root}")
        if unknown:
            print(f"  chords not in base list (their images are skipped): {unknown}")

        for split in SPLITS:
            img_dir, lbl_dir = root / split / "images", root / split / "labels"
            if not img_dir.is_dir():
                print(f"  {split}: no images folder, skipped")
                continue
            dest_split = "valid" if (tag != "base" and split == "test" and args.keep_base_test) else split
            out_img = args.out / dest_split / "images"
            out_lbl = args.out / dest_split / "labels"
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            kept = skipped = 0
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                label = lbl_dir / f"{img.stem}.txt"
                lines = convert_label(label, remap) if label.is_file() else None
                if lines is None:
                    skipped += 1
                    continue
                stem = f"{tag}_{img.stem}"  # prefix avoids filename clashes between datasets
                shutil.copy2(img, out_img / f"{stem}{img.suffix}")
                (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                kept += 1
            arrow = f" -> {dest_split}" if dest_split != split else ""
            print(f"  {split}{arrow}: kept {kept}, skipped {skipped}")

    for split in SPLITS:  # the training code expects all three splits to exist
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)

    data_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(names),
        "names": names,
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"\nMerged dataset written to {args.out} with {len(names)} chords: {names}")

if __name__ == "__main__":
    main()
