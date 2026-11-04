"""Train, evaluate, or predict using the CNN baseline."""

import argparse
import json
from pathlib import Path

from . import CHORDS
from .data import split_dataset


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
    # Validate placeholders before loading the ML runtime.
    if args.command == "train":
        try:
            splits = split_dataset(args.data, args.seed)
        except ValueError as exc:
            parser.error(str(exc))
    elif not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}; train a model first")

    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image, ImageEnhance, ImageOps
    from .model import build_model
    from .metrics import classification_metrics

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = 128

    def image_tensor(path, augment=False):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").resize((size, size))
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
            return image_tensor(path, self.augment), label

    model = build_model().to(device)

    def evaluate(samples):
        model.eval()
        actual, predicted = [], []
        with torch.inference_mode():
            for images, labels in DataLoader(Images(samples), batch_size=args.batch_size):
                actual.extend(labels.tolist())
                predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
        return classification_metrics(actual, predicted, CHORDS)

    if args.command == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        loader = DataLoader(Images(splits[0], augment=True), batch_size=args.batch_size, shuffle=True)
        best = -1.0
        for epoch in range(args.epochs):
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
            if metrics["accuracy"] > best:
                best = metrics["accuracy"]
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "classes": list(CHORDS),
                            "seed": args.seed, "image_size": size, "splits": splits}, args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if checkpoint["classes"] != list(CHORDS) or checkpoint["image_size"] != size:
            parser.error("Checkpoint does not match this model's classes or image size")
        model.load_state_dict(checkpoint["state_dict"])
        if args.command == "evaluate":
            print(json.dumps(evaluate(checkpoint["splits"][2]), indent=2))
        else:
            model.eval()
            with torch.inference_mode():
                probabilities = model(image_tensor(args.image).unsqueeze(0).to(device)).softmax(1)[0]
            index = probabilities.argmax().item()
            print(json.dumps({"chord": CHORDS[index], "confidence": probabilities[index].item()}))


if __name__ == "__main__":
    main()
