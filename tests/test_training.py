import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import EnhancementCNN
from train import run_epoch, train_model


def _loader():
    inputs = torch.rand(2, 3, 8, 8)
    targets = torch.rand(2, 3, 8, 8)
    return DataLoader(TensorDataset(inputs, targets), batch_size=1)


def test_training_saves_best_validation_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "nested" / "best.pt"
    history = train_model(
        EnhancementCNN(features=2),
        _loader(),
        _loader(),
        epochs=2,
        learning_rate=0.001,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert len(history) == 2
    assert checkpoint["epoch"] == min(
        history, key=lambda row: row["validation_loss"]
    )["epoch"]
    assert checkpoint["validation_loss"] == min(
        row["validation_loss"] for row in history
    )


def test_empty_loader_raises_clear_error():
    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0, 3, 8, 8))
    )

    with pytest.raises(ValueError, match="empty data loader"):
        run_epoch(
            EnhancementCNN(features=2),
            empty_loader,
            torch.nn.L1Loss(),
            torch.device("cpu"),
        )
