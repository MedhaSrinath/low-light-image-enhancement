import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import EnhancementCNN
from preprocessing.dataset_utils import calculate_metrics
from preprocessing.paired_dataset import PairedImageDataset


METRIC_FIELDS = (
    "input_PSNR",
    "input_SSIM",
    "enhanced_PSNR",
    "enhanced_SSIM",
)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}. "
            "Train the model first with python train.py."
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model weights: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate_dataset(model, dataset, device, preview_count=3):
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    previews = []

    with torch.inference_mode():
        for index, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs).clamp(0, 1)

            input_image = inputs[0].cpu().permute(1, 2, 0).numpy()
            target_image = targets[0].cpu().permute(1, 2, 0).numpy()
            enhanced_image = predictions[0].cpu().permute(1, 2, 0).numpy()
            filename = dataset.pairs[index][0].name

            input_metrics = calculate_metrics(target_image, input_image)
            enhanced_metrics = calculate_metrics(target_image, enhanced_image)
            results.append(
                {
                    "filename": filename,
                    "input_PSNR": input_metrics["PSNR"],
                    "input_SSIM": input_metrics["SSIM"],
                    "enhanced_PSNR": enhanced_metrics["PSNR"],
                    "enhanced_SSIM": enhanced_metrics["SSIM"],
                }
            )

            if len(previews) < preview_count:
                previews.append(
                    (filename, input_image, target_image, enhanced_image)
                )

    averages = {
        field: float(np.mean([row[field] for row in results]))
        for field in METRIC_FIELDS
    }
    return results, averages, previews


def save_results(results, averages, previews, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "test_metrics.csv"
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("filename", *METRIC_FIELDS))
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({"filename": "AVERAGE", **averages})

    if previews:
        figure, axes = plt.subplots(
            len(previews), 3, figsize=(12, 4 * len(previews)), squeeze=False
        )
        for row, (filename, input_image, target_image, enhanced_image) in enumerate(
            previews
        ):
            for column, (image, label) in enumerate(
                (
                    (input_image, "Low-light input"),
                    (target_image, "Ground truth"),
                    (enhanced_image, "Enhanced output"),
                )
            ):
                axes[row, column].imshow(image)
                axes[row, column].set_title(f"{label}: {filename}")
                axes[row, column].axis("off")

        figure.tight_layout()
        figure.savefig(output_dir / "test_comparisons.png", dpi=150)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model on the held-out test set."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=256)
    args = parser.parse_args()

    if args.image_width < 1 or args.image_height < 1:
        parser.error("--image-width and --image-height must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(EnhancementCNN(), args.checkpoint, device)
    test_data = PairedImageDataset(
        "Test", size=(args.image_width, args.image_height)
    )
    results, averages, previews = evaluate_dataset(model, test_data, device)
    save_results(results, averages, previews, args.output_dir)

    print(f"Evaluated {len(results)} test images on {device}.")
    for metric, value in averages.items():
        print(f"Average {metric}: {value:.4f}")
    print(f"Metrics and comparisons saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
