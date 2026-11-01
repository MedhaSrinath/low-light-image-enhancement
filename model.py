import torch
from torch import nn


MODEL_VERSION = 2


class EnhancementCNN(nn.Module):
    """A small residual CNN that predicts corrections to an RGB image."""

    def __init__(self, features=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, image):
        correction = torch.tanh(self.layers(image))
        return torch.clamp(image + correction, 0, 1)


if __name__ == "__main__":
    model = EnhancementCNN()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {parameter_count:,}")
