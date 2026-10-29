import torch
from torch import nn


class EnhancementCNN(nn.Module):
    """A small CNN that maps RGB images to enhanced RGB images."""

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
            nn.Sigmoid(),
        )

    def forward(self, image):
        return self.layers(image)


if __name__ == "__main__":
    model = EnhancementCNN()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {parameter_count:,}")
