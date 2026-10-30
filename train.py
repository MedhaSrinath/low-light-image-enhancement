import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import EnhancementCNN
from preprocessing.paired_dataset import create_train_validation_datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, phase=""):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_images = 0
    progress_interval = max(1, len(loader) // 5)

    with torch.enable_grad() if training else torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad()

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_images += inputs.size(0)

            if phase and (
                batch_index % progress_interval == 0
                or batch_index == len(loader)
            ):
                print(
                    f"  {phase}: batch {batch_index}/{len(loader)}",
                    flush=True,
                )

    if total_images == 0:
        raise ValueError("Cannot run an epoch with an empty data loader.")

    return total_loss / total_images


def train_model(
    model,
    train_loader,
    validation_loader,
    epochs,
    learning_rate,
    device,
    checkpoint_path,
):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    best_validation_loss = float("inf")
    history = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs}", flush=True)
        training_loss = run_epoch(
            model, train_loader, criterion, device, optimizer, phase="Training"
        )
        validation_loss = run_epoch(
            model, validation_loader, criterion, device, phase="Validation"
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} - "
            f"training loss: {training_loss:.4f}, "
            f"validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )

    return history


def save_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("epoch", "training_loss", "validation_loss")
        )
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    plt.plot(epochs, [row["training_loss"] for row in history], label="Training")
    plt.plot(
        epochs,
        [row["validation_loss"] for row in history],
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("L1 loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train the CNN on paired low-light images."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.image_width < 1 or args.image_height < 1:
        parser.error("--image-width and --image-height must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_data, validation_data = create_train_validation_datasets(
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        size=(args.image_width, args.image_height),
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    history = train_model(
        EnhancementCNN(),
        train_loader,
        validation_loader,
        args.epochs,
        args.learning_rate,
        device,
        args.checkpoint,
    )
    save_history(history, args.output_dir)
    print(f"Best checkpoint saved to: {Path(args.checkpoint).resolve()}")
    print(f"Training history and loss plot saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
