from __future__ import annotations

from typing import Dict

from torchvision import transforms


def build_transforms(cfg: Dict) -> Dict[str, transforms.Compose]:
    size = cfg.get("image_size", 512)
    # Optional anti-overlap crop: remove a fixed fraction of the border before resizing.
    # This is useful when adjacent FOVs overlap (e.g., 5%).
    #
    # Config options (all optional):
    #   crop_border_fraction: float in [0, 0.49]  (e.g., 0.05 removes 5% on each side)
    #   crop_border_pixels: int >= 0             (alternative to fraction)
    # If both are provided, pixels take precedence.
    crop_border_fraction = cfg.get("crop_border_fraction", 0.0)
    crop_border_pixels = cfg.get("crop_border_pixels", None)
    mean = cfg.get("mean", [0.5, 0.5, 0.5])
    std = cfg.get("std", [0.25, 0.25, 0.25])

    def _maybe_center_crop_ops():
        ops = []
        if crop_border_pixels is not None:
            px = int(crop_border_pixels)
            if px < 0:
                raise ValueError("crop_border_pixels must be >= 0")
            if px > 0:
                # CenterCrop expects output size.
                # Use a lambda so it works for arbitrary input sizes.
                ops.append(
                    transforms.Lambda(
                        lambda img, px=px: transforms.functional.center_crop(
                            img, [max(1, img.size[1] - 2 * px), max(1, img.size[0] - 2 * px)]
                        )
                    )
                )
        else:
            frac = float(crop_border_fraction or 0.0)
            if frac < 0 or frac >= 0.5:
                raise ValueError("crop_border_fraction must be in [0, 0.5)")
            if frac > 0:
                ops.append(
                    transforms.Lambda(
                        lambda img, frac=frac: transforms.functional.center_crop(
                            img,
                            [
                                max(1, int(round(img.size[1] * (1 - 2 * frac)))),
                                max(1, int(round(img.size[0] * (1 - 2 * frac)))),
                            ],
                        )
                    )
                )
        return ops

    train_aug = [*_maybe_center_crop_ops(), transforms.Resize((size, size))]
    if cfg.get("random_flip", True):
        train_aug.append(transforms.RandomHorizontalFlip())
        train_aug.append(transforms.RandomVerticalFlip())
    if cfg.get("random_rotation", True):
        train_aug.append(transforms.RandomRotation(15))
    if cfg.get("color_jitter", False):
        train_aug.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    train_aug.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    eval_aug = [*_maybe_center_crop_ops(), transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return {
        "train": transforms.Compose(train_aug),
        "val": transforms.Compose(eval_aug),
        "test": transforms.Compose(eval_aug),
    }
