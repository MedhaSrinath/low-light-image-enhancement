import numpy as np
import pytest
import torch
from PIL import Image

from preprocessing import dataset_utils
from preprocessing.paired_dataset import (
    PairedImageDataset,
    create_train_validation_datasets,
)


def _create_image(path, value):
    image = np.full((4, 5, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def _create_split(root, split, names):
    input_dir = root / split / "Input"
    gt_dir = root / split / "GT"
    input_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    for index, name in enumerate(names):
        _create_image(input_dir / name, index)
        _create_image(gt_dir / name, index + 10)


def test_dataset_loads_filename_matched_pair_as_rgb_tensors(tmp_path, monkeypatch):
    _create_split(tmp_path, "Train", ["b.png", "a.png"])
    monkeypatch.setattr(dataset_utils, "DATA_DIR", tmp_path)

    dataset = PairedImageDataset()
    input_image, gt_image = dataset[0]

    assert len(dataset) == 2
    assert input_image.shape == (3, 4, 5)
    assert gt_image.shape == (3, 4, 5)
    assert input_image.dtype == torch.float32
    assert torch.allclose(input_image, torch.full((3, 4, 5), 1 / 255))
    assert torch.allclose(gt_image, torch.full((3, 4, 5), 11 / 255))


def test_unmatched_images_raise_clear_error(tmp_path, monkeypatch):
    _create_split(tmp_path, "Train", ["paired.png"])
    _create_image(tmp_path / "Train" / "Input" / "missing.png", 0)
    monkeypatch.setattr(dataset_utils, "DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="Unmatched image filenames"):
        PairedImageDataset()


def test_validation_split_is_reproducible_and_uses_train_only(
    tmp_path, monkeypatch
):
    names = [f"{index:02}.png" for index in range(10)]
    _create_split(tmp_path, "Train", names)
    _create_split(tmp_path, "Test", ["test.png"])
    monkeypatch.setattr(dataset_utils, "DATA_DIR", tmp_path)

    training, validation = create_train_validation_datasets(seed=7)
    training_again, validation_again = create_train_validation_datasets(seed=7)

    assert training.dataset.split == "Train"
    assert validation.dataset is training.dataset
    assert len(training) == 8
    assert len(validation) == 2
    assert training.indices == training_again.indices
    assert validation.indices == validation_again.indices
    assert sorted(training.indices + validation.indices) == list(range(10))


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_validation_fraction_must_be_between_zero_and_one(fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        create_train_validation_datasets(validation_fraction=fraction)
