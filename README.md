# Guitar-Chord-Detection-Computer-Vision-Model
Classifies static guitar photos into 11 chords and returns a confidence score.

## Current baseline

The supplied YOLO export uses a pretrained ResNet-18 with a frozen backbone.
Training fits the 11-class head using cross-entropy and saves the checkpoint
with the highest validation macro F1. All splits and prediction use the same
224×224 ResNet-18 preprocessing. Evaluation reports test accuracy and
macro and per-class precision, recall, and F1. Prediction returns the top
chord and its softmax score.

Hand tracking, fretboard localization, webcam capture, and correctness feedback
are future work. This baseline classifies whole images; its confidence is not
calibrated and does not establish correct finger placement.

## Dataset

Data set was downloaded from https://universe.roboflow.com/zhenyuandaxian/detection-model-for-guitar-chord/dataset/6. 

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

Extract the downloaded dataset under `data/raw`. See
[data/README.md](data/README.md) for the required folder layout.

Run from the repository root after adding images:

```sh
guitar-chords train --data data/raw --epochs 10
guitar-chords evaluate --checkpoint artifacts/model.pt
guitar-chords predict --checkpoint artifacts/model.pt --image path/to/image.jpg
```

Training selects CUDA when available and otherwise uses CPU. The first training
run downloads the pretrained backbone weights. Checkpoints and raw
data stay out of Git. Keep the original dataset paths available for test evaluation.

Code lives in `src/guitar_chords/`; dependency-free split and metric tests live
in `tests/`. `notebooks/` is reserved for exploratory analysis.
