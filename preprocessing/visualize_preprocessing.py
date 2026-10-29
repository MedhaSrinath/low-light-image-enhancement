
import matplotlib.pyplot as plt

from preprocessing.dataset_utils import load_dataset, get_image_pairs


if __name__ == "__main__":
    # Load the first three paired training images
    pairs = get_image_pairs("Train")[:3]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    for row, (input_path, gt_path) in enumerate(pairs):
        low_light = load_dataset  # The function is imported below for use
        from PIL import Image

        with Image.open(input_path) as image:
            input_image = image.convert("RGB").resize((256, 256))

        with Image.open(gt_path) as image:
            gt_image = image.convert("RGB").resize((256, 256))

        axes[row, 0].imshow(input_image)
        axes[row, 0].set_title("Low-light Input")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_image)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_visualization.png", dpi=150)
    plt.show()

    print("Visualization saved as preprocessing_visualization.png")
