"""Train, evaluate, or predict guitar chords from static images."""

import argparse
import json
from pathlib import Path

from . import CHORDS
from .data import ChordImageDataset, load_yolo_splits, read_rgb_image, resnet18_preprocessing, split_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "predict"))
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.command == "predict" and args.image is None:
        parser.error("predict requires --image")
    if args.command == "train":
        supplied_splits = (args.data / "data.yaml").is_file()
        try:
            splits = (load_yolo_splits(args.data) if supplied_splits
                      else split_dataset(args.data, args.seed))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    elif not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}; train a model first")

    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import ImageEnhance
    from .model import RESNET18, RESNET18_PREPROCESSING, build_model, build_resnet18
    from .metrics import classification_metrics

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    if args.command != "train":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        architecture = checkpoint.get("architecture", "small_cnn")
        if architecture not in (RESNET18, "small_cnn"):
            parser.error(f"Unsupported checkpoint architecture: {architecture}")
        supplied_splits = architecture == RESNET18
        if supplied_splits and checkpoint.get("preprocessing") != RESNET18_PREPROCESSING:
            parser.error("Unsupported ResNet-18 checkpoint preprocessing")
    size = 224 if supplied_splits else 128

    def legacy_image_tensor(path, augment=False):
        image = read_rgb_image(path).resize((size, size))
        if augment:
            image = image.rotate(float(torch.rand(1).item() * 10 - 5))
            image = ImageEnhance.Brightness(image).enhance(.9 + torch.rand(1).item() * .2)
        return torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(size, size, 3).permute(2, 0, 1).float() / 255

    class Images(Dataset):
        def __init__(self, samples, augment=False):
            self.samples, self.augment = samples, augment

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            path, label = self.samples[index]
            return legacy_image_tensor(path, self.augment), label

    dataset = ChordImageDataset if supplied_splits else Images
    model = (build_resnet18(pretrained=args.command == "train") if supplied_splits
             else build_model()).to(device)

    def evaluate(samples):
        model.eval()
        actual, predicted = [], []
        with torch.inference_mode():
            for images, labels in DataLoader(dataset(samples), batch_size=args.batch_size):
                actual.extend(labels.tolist())
                predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
        return classification_metrics(actual, predicted, CHORDS)

    if args.command == "train":
        trainable = model.fc.parameters() if supplied_splits else model.parameters()
        optimizer = torch.optim.Adam(trainable, lr=.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        training_data = dataset(splits[0]) if supplied_splits else Images(splits[0], augment=True)
        loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
        selection_metric = "macro_f1" if supplied_splits else "accuracy"
        best = -1.0
        for epoch in range(args.epochs):
            # Freeze the backbone's batch normalization statistics during head training.
            if supplied_splits:
                model.eval()
                model.fc.train()
            else:
                model.train()
            total = 0.0
            for images, labels in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(images.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
                total += loss.item() * len(labels)
            metrics = evaluate(splits[1])
            print(json.dumps({"epoch": epoch + 1, "train_loss": total / len(splits[0]), "validation": metrics}))
            if metrics[selection_metric] > best:
                best = metrics[selection_metric]
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = {"state_dict": model.state_dict(), "classes": list(CHORDS),
                              "seed": args.seed, "image_size": size, "splits": splits}
                if supplied_splits:
                    checkpoint.update(architecture=RESNET18, preprocessing=RESNET18_PREPROCESSING)
                torch.save(checkpoint, args.checkpoint)
    else:
        if checkpoint["classes"] != list(CHORDS) or checkpoint["image_size"] != size:
            parser.error("Checkpoint does not match this model's classes or image size")
        model.load_state_dict(checkpoint["state_dict"])
        if args.command == "evaluate":
            print(json.dumps(evaluate(checkpoint["splits"][2]), indent=2))
        else:
            model.eval()
            image = (resnet18_preprocessing()(read_rgb_image(args.image)) if supplied_splits
                     else legacy_image_tensor(args.image))
            with torch.inference_mode():
                probabilities = model(image.unsqueeze(0).to(device)).softmax(1)[0]
            index = probabilities.argmax().item()
            print(json.dumps({"chord": CHORDS[index], "confidence": probabilities[index].item()}))


if __name__ == "__main__":
    main()
