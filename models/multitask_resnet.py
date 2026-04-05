"""
Multi-Task ResNet for joint classification and regression.

Goal: Given a cell image, predict:
  1. Is it infected or uninfected? (Classification)
  2. What is the time? (Regression)
     - For infected: Time since infection onset
     - For uninfected: Elapsed time from experiment start

Architecture:
    Input Image → ResNet Backbone (shared features)
                  ├─> Classification Head → Binary logits [infected/uninfected]
                  └─> Regression Head → Time prediction [scalar]

The shared backbone allows both tasks to benefit from learned representations.
"""
from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
from torchvision import models


class MultiTaskResNet(nn.Module):
    """
    Multi-task learning model with shared backbone and two task-specific heads.
    
    The model predicts both infection status (classification) and time (regression)
    simultaneously, allowing the tasks to help each other through shared features.
    """
    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] = "resnet50",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.2,
        train_backbone: bool = True,
        hidden_dim: int = 256,  # Hidden layer size for task heads
        use_cls_conditioning: bool = False,  # NEW: Use classification info in regression head
    ) -> None:
        super().__init__()
        
        self.use_cls_conditioning = use_cls_conditioning
        self.num_classes = num_classes
        
        # Build ResNet backbone (feature extractor)
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
        else:
            raise AttributeError("Unexpected ResNet architecture without fc layer")
        
        # Replace final FC with identity to extract features
        self.backbone.fc = nn.Identity()
        
        # ============ Classification Head ============
        # Predicts: infected (1) or uninfected (0)
        cls_layers = []
        if dropout > 0:
            cls_layers.append(nn.Dropout(dropout))
        
        if hidden_dim > 0:
            # Two-layer head with ReLU activation
            cls_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_classes)
            ])
        else:
            # Single linear layer
            cls_layers.append(nn.Linear(in_features, num_classes))
        
        self.classifier = nn.Sequential(*cls_layers)
        
        # ============ Regression Head ============
        # Predicts: time (hours)
        #   - For infected: time since infection onset
        #   - For uninfected: elapsed time from experiment start
        #
        # NEW: Can optionally use classification logits as additional input
        reg_input_dim = in_features
        if use_cls_conditioning:
            # Add num_classes dimensions for classification logits
            reg_input_dim += num_classes
        
        reg_layers = []
        if dropout > 0:
            reg_layers.append(nn.Dropout(dropout))
        
        if hidden_dim > 0:
            # Two-layer head with ReLU activation
            reg_layers.extend([
                nn.Linear(reg_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, 1)  # Single output: time
            ])
        else:
            # Single linear layer
            reg_layers.append(nn.Linear(reg_input_dim, 1))
        
        self.regressor = nn.Sequential(*reg_layers)
        
        # Freeze backbone if requested
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing both classification and regression outputs.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            cls_logits: Classification logits [batch_size, num_classes]
            time_pred: Time predictions [batch_size, 1]
        """
        # Shared feature extraction
        features = self.backbone(x)
        
        # Classification prediction (always computed first)
        cls_logits = self.classifier(features)  # [B, 2] for binary classification
        
        # Regression prediction
        if self.use_cls_conditioning:
            # Concatenate features with classification logits
            # This allows regression head to "know" the predicted class
            reg_input = torch.cat([features, cls_logits], dim=1)  # [B, in_features + num_classes]
            time_pred = self.regressor(reg_input)
        else:
            # Standard: only use features
            time_pred = self.regressor(features)  # [B, 1] for time prediction
        
        return cls_logits, time_pred
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features (useful for analysis/visualization)."""
        return self.backbone(x)


def build_multitask_model(cfg: dict) -> nn.Module:
    """
    Build multi-task model from config dictionary.
    
    Config options:
        name: ResNet variant (resnet18/34/50/101/152)
        pretrained: Use ImageNet pretrained weights
        num_classes: Number of classification classes (typically 2)
        dropout: Dropout rate
        train_backbone: Whether to fine-tune backbone
        hidden_dim: Hidden layer size for task heads (0 = no hidden layer)
        use_cls_conditioning: Whether regression head receives classification logits (default: False)
    
    Example config:
        model:
          name: resnet50
          pretrained: true
          num_classes: 2
          dropout: 0.2
          train_backbone: true
          hidden_dim: 256
          use_cls_conditioning: true  # NEW: Enable classification conditioning
    """
    return MultiTaskResNet(
        backbone=cfg.get("name", "resnet50"),
        pretrained=cfg.get("pretrained", True),
        num_classes=cfg.get("num_classes", 2),
        dropout=cfg.get("dropout", 0.2),
        train_backbone=cfg.get("train_backbone", True),
        hidden_dim=cfg.get("hidden_dim", 256),
        use_cls_conditioning=cfg.get("use_cls_conditioning", False),
    )
