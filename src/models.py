from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import timm

from src.settings import DEVICE, UNET_CHECKPOINT, VIT_CHECKPOINT


def resolve_device() -> torch.device:
    if DEVICE:
        return torch.device(DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CheckpointUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_segmentation_model(
    checkpoint_path: str | None = None,
    device: torch.device | None = None,
) -> tuple[CheckpointUNet, torch.device]:
    device = device or resolve_device()
    path = str(checkpoint_path or UNET_CHECKPOINT)
    model = CheckpointUNet(in_channels=3, out_channels=1)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device


def load_classifier_model(
    checkpoint_path: str | None = None,
    device: torch.device | None = None,
) -> tuple[ViTClassifier, dict[int, str], dict[str, int], dict[str, Any], torch.device]:
    device = device or resolve_device()
    path = str(checkpoint_path or VIT_CHECKPOINT)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
    class_to_idx = {str(k): int(v) for k, v in checkpoint["class_to_idx"].items()}
    model = ViTClassifier(num_classes=len(idx_to_class), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    meta = {
        "best_val_acc": float(checkpoint.get("best_val_acc", 0.0)),
        "epoch": int(checkpoint.get("epoch", -1)),
    }
    return model, idx_to_class, class_to_idx, meta, device
