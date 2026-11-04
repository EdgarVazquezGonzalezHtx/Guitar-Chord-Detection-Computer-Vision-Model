# Guitar-Chord-Detection-Computer-Vision-Model
Computer vision model that uses hand and fretboard tracking to see if a user is playing a chord correctly.

## Current baseline

Python/PyTorch skeleton based on the project proposal: an 11-class CNN,
RGB images resized to 128×128, small training-only rotations and brightness
augmentation, cross-entropy loss, and a reproducible 70/20/10 split.
The best validation-accuracy checkpoint is retained; evaluation uses only its
held-out test split and reports accuracy and macro/per-class precision, recall,
and F1. Prediction returns a chord and a softmax confidence score.

Hand tracking, fretboard localization, webcam capture, and correctness feedback
are future work. This baseline classifies whole images; its confidence is not
calibrated and does not establish correct finger placement.

## Setup

Requires Python 3.10+.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

For a CPU-only machine, install PyTorch first with
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
Jupyter is optional: `python -m pip install -e '.[notebook]'`.

## Dataset and commands

The dataset is intentionally empty. See [data/README.md](data/README.md) for
the required folder layout. No pretrained model or real training results are included.

Run from the repository root after adding images:

```sh
guitar-chords train --data data/raw --epochs 10
guitar-chords evaluate --checkpoint artifacts/model.pt
guitar-chords predict --checkpoint artifacts/model.pt --image path/to/image.jpg
```

Training selects CUDA when available and otherwise uses CPU. Checkpoints and raw
data stay out of Git. Keep the original dataset paths available for test evaluation.

Code lives in `src/guitar_chords/`; dependency-free split and metric tests live
in `tests/`. `notebooks/` is reserved for exploratory analysis.
