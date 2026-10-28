from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


DATA_DIR = Path("data/LOL-v2-real")
SPLIT = "Train"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

input_dir = DATA_DIR / SPLIT / "Input"
gt_dir = DATA_DIR / SPLIT / "GT"

for directory in (input_dir, gt_dir):
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory.resolve()}")

input_files = {
    path.name
    for path in input_dir.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
}
gt_files = {
    path.name
    for path in gt_dir.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
}
matching_files = sorted(input_files & gt_files)
missing_gt = sorted(input_files - gt_files)
missing_input = sorted(gt_files - input_files)

print("Input images:", len(input_files))
print("GT images:", len(gt_files))
print("Matching filenames:", len(matching_files))
print("Missing GT:", missing_gt[:10])
print("Missing Input:", missing_input[:10])

if not matching_files:
    raise RuntimeError("No matching filenames found. Check the dataset structure.")

sample_names = matching_files[:3]
fig, axes = plt.subplots(
    len(sample_names),
    2,
    figsize=(10, 4 * len(sample_names)),
    squeeze=False,
)

for row, name in enumerate(sample_names):
    for column, (folder, label) in enumerate(
        ((input_dir, "Input (low-light)"), (gt_dir, "GT (reference)"))
    ):
        image_path = folder / name
        with Image.open(image_path) as image:
            displayed_image = image.convert("RGB")
        axes[row, column].imshow(displayed_image)
        axes[row, column].set_title(f"{label}: {name} ({displayed_image.width}x{displayed_image.height})")
        axes[row, column].axis("off")

fig.tight_layout()
preview_path = DATA_DIR / f"{SPLIT.lower()}_preview.png"
fig.savefig(preview_path, dpi=150, bbox_inches="tight")
print("Preview saved to:", preview_path.resolve())
plt.show()
