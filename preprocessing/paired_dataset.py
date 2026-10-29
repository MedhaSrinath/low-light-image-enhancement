import torch
from torch.utils.data import Dataset, random_split

from preprocessing.dataset_utils import get_image_pairs, load_image


class PairedImageDataset(Dataset):
    """Load one low-light image and its reference on demand."""

    def __init__(self, split="Train", size=None):
        self.split = split
        self.size = size
        self.pairs = get_image_pairs(split)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, gt_path = self.pairs[index]
        input_image = load_image(input_path, size=self.size)
        gt_image = load_image(gt_path, size=self.size)

        if input_image.shape != gt_image.shape:
            raise ValueError(
                f"Paired images have different dimensions: "
                f"{input_path.name} {input_image.shape} != "
                f"{gt_path.name} {gt_image.shape}"
            )

        input_tensor = torch.from_numpy(input_image).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(gt_image).permute(2, 0, 1)
        return input_tensor, gt_tensor


def create_train_validation_datasets(validation_fraction=0.2, seed=42, size=None):
    """Split only the training images into reproducible train and validation sets."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    dataset = PairedImageDataset("Train", size=size)
    validation_size = int(len(dataset) * validation_fraction)
    training_size = len(dataset) - validation_size
    if validation_size == 0 or training_size == 0:
        raise ValueError("The dataset is too small for the requested split.")

    generator = torch.Generator().manual_seed(seed)
    return random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )
