import numpy as np
import pytest
import torch
from PIL import Image

from model import EnhancementCNN
from preprocessing import dataset_utils
from preprocessing.paired_dataset import PairedImageDataset
from evaluate import evaluate_dataset, load_checkpoint, save_results


def _create_split(root):
    input_dir = root / "Test" / "Input"
    gt_dir = root / "Test" / "GT"
    input_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    for name, input_value in (("a.png", 30), ("b.png", 60)):
        input_image = np.full((16, 16, 3), input_value, dtype=np.uint8)
        gt_image = np.full((16, 16, 3), input_value + 20, dtype=np.uint8)
        Image.fromarray(input_image).save(input_dir / name)
        Image.fromarray(gt_image).save(gt_dir / name)


def test_evaluation_reports_metrics_and_saves_results(tmp_path, monkeypatch):
    _create_split(tmp_path)
    monkeypatch.setattr(dataset_utils, "DATA_DIR", tmp_path)
    dataset = PairedImageDataset("Test")

    results, averages, previews = evaluate_dataset(
        EnhancementCNN(features=2), dataset, torch.device("cpu")
    )
    save_results(results, averages, previews, tmp_path / "outputs")

    assert [row["filename"] for row in results] == ["a.png", "b.png"]
    assert set(averages) == {
        "input_PSNR",
        "input_SSIM",
        "enhanced_PSNR",
        "enhanced_SSIM",
    }
    assert len(previews) == 2
    assert (tmp_path / "outputs" / "test_metrics.csv").is_file()
    assert (tmp_path / "outputs" / "test_comparisons.png").is_file()


def test_missing_checkpoint_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the model first"):
        load_checkpoint(
            EnhancementCNN(features=2),
            tmp_path / "missing.pt",
            torch.device("cpu"),
        )
