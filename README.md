# Guitar-Chord-Detection-Computer-Vision-Model
Classifies static guitar photos into 11 chords and returns a confidence score.

## Current baseline

The supplied YOLO export uses a pretrained ResNet-18 with a frozen backbone.
Training fits the 11-class head using cross-entropy and saves the checkpoint
with the highest validation macro F1. All splits and prediction use the same
224×224 ResNet-18 preprocessing. Evaluation reports test accuracy and
macro and per-class precision, recall, and F1. Prediction returns the top
chord and its softmax score.

The `--unfreeze` option fine tunes the whole neural network instead of only the
11 classes. Use it with a small learning rate like : --lr 1e-4. On the 188 image
test set, the frozen baseline reaches 0.915 accuracy, the fine tuned model 0.989,
and the fine tuned model trained with the extra Hugging Face data 1.000.

Hand tracking, fretboard localization, webcam capture, and correctness feedback
are future work. This baseline classifies whole images; its confidence is not
calibrated and does not establish correct finger placement.

## Dataset

Data set was downloaded from https://universe.roboflow.com/zhenyuandaxian/detection-model-for-guitar-chord/dataset/6. 

Optional extra data: https://huggingface.co/datasets/dduka/guitar-chords. Only the
9 chords it shares with the Roboflow set are used.

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

Extract the downloaded dataset (YOLOv8 format) under `data/raw`, so that
`data.yaml` and the `train/`, `valid/`, and `test/` folders sit directly in
`data/raw`.

Run from the repository root after adding images:

```sh
guitar-chords train --data data/raw --epochs 10
guitar-chords evaluate --checkpoint artifacts/model.pt
guitar-chords predict --checkpoint artifacts/model.pt --image path/to/image.jpg
```

To train the fine tuned model:

```sh
guitar-chords train --data data/raw --epochs 65 --lr 1e-4 --unfreeze --checkpoint artifacts/raw_ft.pt
guitar-chords evaluate --checkpoint artifacts/raw_ft.pt
```

To add the Hugging Face data and train on both datasets:

```sh
python -m pip install datasets
python data/new_dataset/huggingface_to_yolo.py dduka/guitar-chords --out data/extra1
python data/new_dataset/mergehf_with_orig.py --base data/raw --add data/extra1 --out data/merged --keep-base-test
guitar-chords train --data data/merged --epochs 65 --lr 1e-4 --unfreeze --checkpoint artifacts/merged_ft.pt
guitar-chords evaluate --checkpoint artifacts/merged_ft.pt
```

`--keep-base-test` keeps the Roboflow test set unchanged, so scores stay comparable.

Training selects CUDA when available and otherwise uses CPU. The first training
run downloads the pretrained backbone weights. Checkpoints and raw
data stay out of Git. Keep the original dataset paths available for test evaluation.

Code lives in `src/guitar_chords/`; dataset scripts live in `data/new_dataset/`;
dependency-free split and metric tests live in `tests/`. `notebooks/` is reserved
for exploratory analysis.
