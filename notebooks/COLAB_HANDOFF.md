# Guitar Chord Classifier Colab Handoff

The Phase 1 model predicts one of 11 guitar chords from a static photo and returns a softmax confidence score.

## Current validation results

The first full run used **10 epochs, batch size 32, and seed 42**. The best saved checkpoint correctly classified **294 of 378 validation images**.

| Accuracy | Macro precision | Macro recall | Macro F1 |
|---:|---:|---:|---:|
| 77.8% | 79.2% | 78.6% | 77.8% |

| Chord | Precision | Recall | F1 |
|---|---:|---:|---:|
| A | 94.4% | 85.0% | 89.5% |
| Am | 76.7% | 89.2% | 82.5% |
| B7 | 86.7% | 89.7% | 88.1% |
| Bm | 92.6% | 78.1% | 84.7% |
| C | 77.8% | 61.8% | 68.9% |
| D | 68.8% | 84.6% | 75.9% |
| D7 | 64.1% | 69.4% | 66.7% |
| Dm | 64.7% | 75.9% | 69.8% |
| Em | 90.9% | 47.6% | 62.5% |
| F | 70.7% | 93.5% | 80.6% |
| G | 83.9% | 89.7% | 86.7% |

Em had the lowest recall. Use its recall alongside macro F1 when comparing runs. The held-out test evaluation is pending.

## Access the shared project

1. Ask the project owner to share the Google Drive project folder. It contains `notebooks/train_colab.ipynb`, `pyproject.toml`, `src/`, `data/raw/`, and saved checkpoints under `artifacts/`.
2. In Drive, add a shortcut to the shared project folder under **My Drive**. The shortcut points to the owner's original folder and uses the permissions they granted. Each teammate mounts their own Drive from their own Colab runtime. Sharing the notebook alone gives access to the notebook file. [Google Drive shortcuts](https://support.google.com/drive/answer/9700156) · [Colab sharing FAQ](https://research.google.com/colaboratory/faq.html#using-colab)
3. Open the current `train_colab.ipynb` in Colab. For independent experiments, save a personal copy of the notebook so teammates can change settings independently. Enable a GPU runtime, run the first cell, and approve Drive access. Set `PROJECT_ROOT` to the shared folder path as it appears in Colab's **Files** panel. The default works if the shortcut is at the root of My Drive with the original project name.
4. If you have Viewer access to the project, change `ARTIFACTS` in the first cell to a writable folder in your own Drive, such as `Path("/content/drive/MyDrive/guitar-chord-runs")`. Use the shared `artifacts/` directory if you have write access and want teammates to see your checkpoints.
5. Run **Install and check the runtime** and confirm it prints a GPU name. For Viewer access, replace the first `subprocess.run(...pip...)` line in that cell with the code below so the package can build in a writable directory.

```python
import shutil

build_root = Path("/content/guitar_chords_build")
build_root.mkdir(parents=True, exist_ok=True)
shutil.copy2(PROJECT_ROOT / "pyproject.toml", build_root / "pyproject.toml")
shutil.copytree(PROJECT_ROOT / "src", build_root / "src", dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", str(build_root)], check=True)
```

## Copy the dataset to Colab's local disk

**No manual upload to Colab is needed.** Copy the existing shared `data/raw/` folder to the runtime's local disk for training. It contains `data.yaml` and the original train (1,314), validation (378), and test (188) splits, each with an `images/` and a `labels/` folder. Colab installs the package during setup and saves checkpoints to Drive.

Add this code cell immediately before **Train and save the best checkpoint** and run it after mounting Drive. It copies the dataset from Drive without requiring a ZIP upload.

```python
import shutil

drive_data = PROJECT_ROOT / "data" / "raw"
local_data = Path("/content/guitar_dataset_staged")
local_data.mkdir(parents=True, exist_ok=True)
shutil.copy2(drive_data / "data.yaml", local_data / "data.yaml")

for split in ("train", "valid", "test"):
    print(f"Copying {split} from Drive...", flush=True)
    shutil.copytree(drive_data / split, local_data / split, dirs_exist_ok=True)

DATA_ROOT = local_data
print(f"Training dataset: {DATA_ROOT}", flush=True)
```

Repeat this cell in each new runtime. Colab discards the temporary copy when the runtime is deleted. [Colab runtime FAQ](https://research.google.com/colaboratory/faq.html#using-colab)

## Run and compare experiments

**Next task for the team is to run experiments with different epoch counts, batch sizes, and random seeds.** Compare epoch counts and batch sizes by varying one setting at a time from the baseline. Repeat promising settings across multiple seeds to see how much the validation scores vary. Keep the dataset splits fixed and select the final setup using validation results.

1. In **Train and save the best checkpoint**, set `EPOCHS`, `BATCH_SIZE`, and `SEED`. The baseline used **10 epochs, batch size 32, and seed 42**. Each run starts from pretrained ResNet-18 weights with a fresh chord classifier head. The seed changes head initialization and training shuffle while the supplied splits stay fixed.
2. Run the training cell and record the settings, best validation macro F1, and checkpoint path. Compare experiments on validation macro F1 and per-chord recall. Change one setting at a time when you want to isolate its effect.
3. Training saves the best checkpoint as `<ARTIFACTS>/<run-id>/model.pt`. The current baseline is at `artifacts/20261124T155953951297Z/model.pt` in the owner's project folder. If an older notebook copy only prints `CompletedProcess`, use the current repository notebook, which streams the training and validation logs.
4. Leave `RUN_TEST_EVALUATION = False` while tuning. Once the team selects a model, set `checkpoint` to the selected `model.pt`, set `RUN_TEST_EVALUATION = True`, and run the final cell. Recreate the same dataset staging path before evaluating a checkpoint in a new runtime because the checkpoint stores the image paths. The saved weights can be evaluated without retraining.
