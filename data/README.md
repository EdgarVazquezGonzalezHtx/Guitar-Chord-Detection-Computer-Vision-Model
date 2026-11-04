# Dataset placeholder

No dataset is included or downloaded. Add licensed JPG/PNG images to
`raw/<chord>/`, using these exact folder names:

`A`, `Am`, `B7`, `Bm`, `C`, `D`, `D7`, `Dm`, `Em`, `F`, `G`.

Provide at least 10 images per class. Raw images are ignored by Git.
Training splits each class 70%/20%/10% with seed 42 (rounded down for
training and validation; the remainder goes to test). Files are never moved.
Split membership is saved in the checkpoint. Keep data paths stable for evaluation.

For a final study, split by guitarist or recording session before training to
avoid leakage between near-duplicate frames. The baseline's random image split
does not enforce subject/session separation.
