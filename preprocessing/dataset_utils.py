
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "LOL-v2-real"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def get_image_pairs(split="Train"):
    """Find matching low-light input and ground-truth image pairs."""

    input_dir = DATA_DIR / split / "Input"
    gt_dir = DATA_DIR / split / "GT"

    if not input_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find Input and GT folders for split: {split}"
        )

    input_files = {
        path.name: path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    gt_files = {
        path.name: path
        for path in gt_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    input_only = sorted(input_files.keys() - gt_files.keys())
    gt_only = sorted(gt_files.keys() - input_files.keys())
    if input_only or gt_only:
        raise ValueError(
            f"Unmatched image filenames in {split}: "
            f"missing GT for {input_only[:5]}, missing Input for {gt_only[:5]}"
        )

    common_names = sorted(input_files.keys())
    if not common_names:
        raise ValueError(f"No images found in the {split} folders.")

    pairs = [
        (input_files[name], gt_files[name])
        for name in common_names
    ]

    print(f"{split} split:")
    print(f"  Input images: {len(input_files)}")
    print(f"  GT images: {len(gt_files)}")
    print(f"  Matching pairs: {len(pairs)}")
    return pairs


def load_image(image_path, size=None):
    """Load an RGB image, optionally resize it, and normalize pixels."""

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if size is not None:
            image = image.resize(size, Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0

    return image_array


def load_dataset(split="Train", size=(256, 256)):
    """Load paired input and reference images into NumPy arrays."""

    pairs = get_image_pairs(split)

    inputs = []
    ground_truths = []

    for input_path, gt_path in pairs:
        inputs.append(load_image(input_path, size=size))
        ground_truths.append(load_image(gt_path, size=size))

    return np.stack(inputs), np.stack(ground_truths)


def calculate_metrics(reference, prediction):
    """Calculate PSNR and SSIM for two RGB images."""

    reference = np.clip(reference, 0.0, 1.0)
    prediction = np.clip(prediction, 0.0, 1.0)

    psnr = peak_signal_noise_ratio(
        reference, prediction, data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        prediction,
        channel_axis=-1,
        data_range=1.0,
    )

    return {"PSNR": float(psnr), "SSIM": float(ssim)}
