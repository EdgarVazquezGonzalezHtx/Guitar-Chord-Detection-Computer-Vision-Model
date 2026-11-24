# Dataset

Place the YOLO export under `raw/` with `data.yaml` and the supplied
`train/`, `valid/`, and `test/` directories. Each split has `images/`
and `labels/` directories. Image and label filenames share a stem.

The ResNet-18 workflow uses these splits as supplied and reads each
image's chord ID from its label file. It applies the pretrained model's
deterministic preprocessing with no added augmentation.

## Class-folder baseline

To use the original small CNN baseline, add licensed JPG/PNG images to
`raw/<chord>/`, using these exact folder names:

`A`, `Am`, `B7`, `Bm`, `C`, `D`, `D7`, `Dm`, `Em`, `F`, `G`.

Provide at least 10 images per class. Raw images are ignored by Git.
Training splits each class 70%/20%/10% with seed 42 (rounded down for
training and validation; the remainder goes to test). Files are never moved.
Split membership is saved in the checkpoint. Keep data paths stable for evaluation.

For a final study, split by guitarist or recording session before training to
avoid leakage between near-duplicate frames. The baseline's random image split
does not enforce subject/session separation.
