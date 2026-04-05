from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] = "resnet50",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.2,
        train_backbone: bool = True,
    ) -> None:
        super().__init__()
        builder = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }[backbone]
        weight_map: Dict[str, models.ResNet50_Weights] = {
            "resnet18": getattr(models, "ResNet18_Weights", None),
            "resnet34": getattr(models, "ResNet34_Weights", None),
            "resnet50": getattr(models, "ResNet50_Weights", None),
            "resnet101": getattr(models, "ResNet101_Weights", None),
            "resnet152": getattr(models, "ResNet152_Weights", None),
        }
        weights_cls = weight_map[backbone]
        weights = weights_cls.IMAGENET1K_V1 if (pretrained and weights_cls is not None) else None
        self.backbone = builder(weights=weights)
        if hasattr(self.backbone, "fc"):
            in_features = self.backbone.fc.in_features
        else:  # pragma: no cover
            raise AttributeError("Unexpected ResNet architecture without fc layer")
        self.backbone.fc = nn.Identity()
        head: list[nn.Module] = []
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*head)
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        features = self.backbone(x)
        return self.classifier(features)


def build_model(cfg: dict) -> nn.Module:
    return ResNetClassifier(
        backbone=cfg.get("name", "resnet50"),
        pretrained=cfg.get("pretrained", True),
        num_classes=cfg.get("num_classes", 2),
        dropout=cfg.get("dropout", 0.2),
        train_backbone=cfg.get("train_backbone", True),
    )
