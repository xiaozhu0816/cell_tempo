from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn as nn
from torchvision import models


class ResNetRegressor(nn.Module):
    """ResNet backbone + regression head producing a single scalar."""

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.2,
        train_backbone: bool = True,
        hidden_dim: int = 256,
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
        if hidden_dim > 0:
            head.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(hidden_dim, 1),
                ]
            )
        else:
            head.append(nn.Linear(in_features, 1))
        self.regressor = nn.Sequential(*head)

        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        feat = self.backbone(x)
        return self.regressor(feat)


def build_regression_model(cfg: dict) -> nn.Module:
    """Build regression-only model from config."""

    return ResNetRegressor(
        backbone=cfg.get("name", "resnet50"),
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.2),
        train_backbone=cfg.get("train_backbone", True),
        hidden_dim=cfg.get("hidden_dim", 256),
    )
