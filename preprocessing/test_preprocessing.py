
from preprocessing.dataset_utils import (
    get_image_pairs,
    load_dataset,
    calculate_metrics,
)

if __name__ == "__main__":
    
    pairs = get_image_pairs("Train")

    # Load the first five training pairs to test preprocessing
    inputs, ground_truths = load_dataset("Train")
    inputs = inputs[:5]
    ground_truths = ground_truths[:5]

    print("\nPreprocessing test")
    print("Input array shape:", inputs.shape)
    print("Ground-truth array shape:", ground_truths.shape)
    print("Minimum pixel value:", inputs.min())
    print("Maximum pixel value:", inputs.max())

    # Compare an image with itself as a metric sanity check
    metrics = calculate_metrics(ground_truths[0], ground_truths[0])
    print("\nMetric sanity check:", metrics)
