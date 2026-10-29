import torch

from model import EnhancementCNN


def test_model_preserves_shape_and_outputs_rgb_values():
    model = EnhancementCNN()
    image = torch.rand(2, 3, 64, 80)

    enhanced = model(image)

    assert enhanced.shape == image.shape
    assert torch.all((enhanced >= 0) & (enhanced <= 1))


def test_model_parameters_receive_gradients():
    model = EnhancementCNN()
    image = torch.rand(1, 3, 32, 32)

    model(image).mean().backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_default_model_parameter_count():
    model = EnhancementCNN()

    assert sum(parameter.numel() for parameter in model.parameters()) == 20_259
