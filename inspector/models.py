"""The models that are compared.

    small_cnn      : 5 convolution blocks, trained from scratch (about 390,000 weights)
    resnet18       : pretrained on ImageNet, then fine-tuned (about 11 million weights)
    resnet18_relit : the same ResNet-18, trained with wider brightness augmentation

Both take a (n, 1, 224, 224) grayscale batch in the range 0 to 1 and return
one number per photo (a logit). sigmoid(logit) = probability the part is defective.
"""

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from inspector import config
from inspector.data import to_tensor


def conv_block(channels_in: int, channels_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(channels_in, channels_out, 3, stride, padding=1, bias=False),
                         nn.BatchNorm2d(channels_out),
                         nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16, stride=2), nn.MaxPool2d(2),     # 224 -> 112 -> 56
            conv_block(16, 32), nn.MaxPool2d(2),              # -> 28
            conv_block(32, 64), nn.MaxPool2d(2),              # -> 14
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, 1))

    @property
    def cam_layer(self) -> nn.Module:
        """Last convolution block (14 x 14 feature map), used by Grad-CAM."""
        return self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - 0.5) / 0.5
        return self.head(self.features(x)).squeeze(1)


class PretrainedResNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)
        # ImageNet models expect 3 colour channels normalised with these values.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def cam_layer(self) -> nn.Module:
        """Last residual stage (7 x 7 feature map), used by Grad-CAM."""
        return self.net.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.net(x).squeeze(1)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN()
    if name in ("resnet18", "resnet18_relit"):
        return PretrainedResNet(pretrained)
    raise ValueError(f"Unknown model: {name}")


def model_path(name: str, fraction: float = 1.0):
    suffix = "" if fraction >= 1 else f"_{round(fraction * 100)}pct"
    return config.MODEL_DIR / f"{name}{suffix}.pt"


def load_model(name: str, fraction: float = 1.0) -> nn.Module:
    """A trained model from the models/ folder, ready to predict."""
    model = build_model(name, pretrained=False)
    model.load_state_dict(torch.load(model_path(name, fraction), weights_only=True))
    return model.eval()


@torch.no_grad()
def predict_proba(model: nn.Module, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """uint8 photos (n, 224, 224) -> probability of "defective" for each."""
    model.eval()
    batches = to_tensor(images).split(batch_size)
    return torch.cat([torch.sigmoid(model(batch)) for batch in batches]).numpy()
