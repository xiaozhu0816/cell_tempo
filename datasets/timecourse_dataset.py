from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

_LABEL_PATTERN = re.compile(r"_p(?P<position>\d+)_")


@dataclass
class FrameExtractionPolicy:
    """Rules for expanding each TIFF stack into multiple frame samples."""

    frames_per_hour: float = 2.0
    infected_window_hours: Tuple[float, float] = (16.0, 30.0)
    uninfected_window_hours: Optional[Tuple[float, float]] = None  # NEW: Optional time window for uninfected
    infected_stride: int = 1
    uninfected_stride: int = 1
    uninfected_use_all: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "FrameExtractionPolicy":
        data = data or {}
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "window_hours" in data:  # backwards compatibility
            filtered["infected_window_hours"] = tuple(data["window_hours"])
        # Handle uninfected_window_hours conversion to tuple
        if "uninfected_window_hours" in data and data["uninfected_window_hours"] is not None:
            filtered["uninfected_window_hours"] = tuple(data["uninfected_window_hours"])
        return cls(**filtered)

    def infected_indices(self, total_frames: int) -> List[int]:
        start_h, end_h = self.infected_window_hours
        start_idx = max(0, math.floor(start_h * self.frames_per_hour))
        end_idx = max(start_idx, math.floor(end_h * self.frames_per_hour))
        end_idx = min(end_idx, total_frames - 1)
        stride = max(1, self.infected_stride)
        indices = list(range(start_idx, end_idx + 1, stride))
        if not indices:
            indices = [min(start_idx, total_frames - 1)]
        return indices

    def uninfected_indices(self, total_frames: int) -> List[int]:
        stride = max(1, self.uninfected_stride)
        
        # If uninfected_window_hours is specified, use the same logic as infected
        if self.uninfected_window_hours is not None:
            start_h, end_h = self.uninfected_window_hours
            start_idx = max(0, math.floor(start_h * self.frames_per_hour))
            end_idx = max(start_idx, math.floor(end_h * self.frames_per_hour))
            end_idx = min(end_idx, total_frames - 1)
            indices = list(range(start_idx, end_idx + 1, stride))
            if not indices:
                indices = [min(start_idx, total_frames - 1)]
            return indices
        
        # Otherwise, use the old behavior
        if not self.uninfected_use_all:
            return [0]
        return list(range(0, total_frames, stride))


_SPLIT_KEY_ALIASES = {
    "train": ("train",),
    "val": ("val", "validation"),
    "test": ("test", "eval", "evaluation"),
}
_SPECIAL_FRAME_KEYS = set().union(*_SPLIT_KEY_ALIASES.values(), {"default"})


def _policy_config_for_split(frames_cfg: Optional[Dict], split_name: str) -> Dict:
    frames_cfg = frames_cfg or {}
    split_name = split_name.lower()
    has_nested = any(key in frames_cfg for key in _SPECIAL_FRAME_KEYS)
    if not has_nested:
        return frames_cfg
    base_cfg = {k: v for k, v in frames_cfg.items() if k not in _SPECIAL_FRAME_KEYS}
    overrides: Optional[Dict] = None
    for key in _SPLIT_KEY_ALIASES.get(split_name, (split_name,)):
        value = frames_cfg.get(key)
        if isinstance(value, dict):
            overrides = value
            break
    if overrides is None and split_name != "train":
        train_cfg = frames_cfg.get("train")
        if isinstance(train_cfg, dict):
            overrides = train_cfg
    if overrides is None:
        default_cfg = frames_cfg.get("default")
        if isinstance(default_cfg, dict):
            overrides = default_cfg
    merged = {**base_cfg}
    if overrides:
        merged.update(overrides)
    return merged


def _resolve_policy(frames_cfg: Optional[Dict], split_name: str) -> FrameExtractionPolicy:
    return FrameExtractionPolicy.from_dict(_policy_config_for_split(frames_cfg, split_name))


def resolve_frame_policies(frames_cfg: Optional[Dict]) -> Dict[str, FrameExtractionPolicy]:
    return {split: _resolve_policy(frames_cfg, split) for split in ("train", "val", "test")}


def format_policy_summary(policy: FrameExtractionPolicy) -> str:
    start, end = policy.infected_window_hours
    
    # Format uninfected info
    if policy.uninfected_window_hours is not None:
        u_start, u_end = policy.uninfected_window_hours
        uninfected_info = f"[{u_start:.1f},{u_end:.1f}]"
    else:
        uninfected_info = "all" if policy.uninfected_use_all else "first-only"
    
    return (
        f"frames_per_hour={policy.frames_per_hour:.2f} | "
        f"infected=[{start:.1f},{end:.1f}] stride={policy.infected_stride} | "
        f"uninfected={uninfected_info} stride={policy.uninfected_stride}"
    )


@dataclass
class DataItem:
    path: Path
    label: int
    condition: str
    position: str
    total_frames: int


@dataclass
class FrameSample:
    path: Path
    label: int
    condition: str
    position: str
    frame_index: int
    total_frames: int
    hours_since_start: float


@dataclass
class DataSplit:
    train: List[DataItem]
    val: List[DataItem]
    test: List[DataItem]


class TimeCourseTiffDataset(Dataset):
    """Torch dataset for multi-frame TIFF stacks coming from live-cell imaging."""

    def __init__(
        self,
        samples: Sequence[FrameSample],
        transform: Optional[Callable] = None,
        frames_per_hour: float = 2.0,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.frames_per_hour = frames_per_hour if frames_per_hour > 0 else 1.0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frame = self._load_frame(sample)
        image = self._to_image(frame)
        if self.transform:
            image = self.transform(image)
        hours = getattr(sample, "hours_since_start", None)
        if hours is None:
            hours = sample.frame_index / self.frames_per_hour
        return (
            image,
            sample.label,
            {
                "path": str(sample.path),
                "condition": sample.condition,
                "position": sample.position,
                "frame_index": sample.frame_index,
                "hours_since_start": float(hours),
            },
        )

    def _load_frame(self, sample: FrameSample) -> np.ndarray:
        with tifffile.TiffFile(sample.path) as tif:
            index = max(0, min(sample.total_frames - 1, sample.frame_index))
            frame = tif.asarray(key=index)
        return frame.astype(np.float32)

    def _to_image(self, frame: np.ndarray) -> Image.Image:
        frame = frame - frame.min()
        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        return Image.fromarray(frame)

    def get_metadata(self, idx: int) -> dict:
        """
        Get metadata for a sample without loading the image.
        Used for temporal generalization analysis.
        """
        sample = self.samples[idx]
        hours = getattr(sample, "hours_since_start", None)
        if hours is None:
            hours = sample.frame_index / self.frames_per_hour
        return {
            "path": str(sample.path),
            "condition": sample.condition,
            "position": sample.position,
            "frame_index": sample.frame_index,
            "hours_since_start": float(hours),
            "label": sample.label,
        }


def _scan_condition(condition_dir: Path, label: int, condition_name: str) -> List[DataItem]:
    condition_dir = Path(condition_dir)
    if not condition_dir.exists():
        raise FileNotFoundError(f"Condition directory not found: {condition_dir}")
    items: List[DataItem] = []
    for path in sorted(condition_dir.glob("*.tif*")):
        match = _LABEL_PATTERN.search(path.name)
        position = match.group("position") if match else "unknown"
        with tifffile.TiffFile(path) as tif:
            total_frames = len(tif.pages)
        items.append(
            DataItem(path=path, label=label, condition=condition_name, position=position, total_frames=total_frames)
        )
    if not items:
        raise RuntimeError(f"No TIFF files discovered in {condition_dir}")
    return items


def _stratified_split(items: Sequence[DataItem], ratios: Sequence[float], seed: int = 42) -> DataSplit:
    if len(ratios) != 3:
        raise ValueError("Split ratios must contain three values (train/val/test)")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-2):
        raise ValueError("Split ratios must sum to 1.0")
    label_buckets: Dict[int, List[DataItem]] = {}
    for item in items:
        label_buckets.setdefault(item.label, []).append(item)
    rng = random.Random(seed)
    for bucket in label_buckets.values():
        rng.shuffle(bucket)
    train, val, test = [], [], []
    for bucket in label_buckets.values():
        n = len(bucket)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])
    return DataSplit(train=train, val=val, test=test)


def build_datasets(
    data_cfg: Dict,
    transforms: Dict[str, Optional[Callable]],
    fold_index: Optional[int] = None,
    num_folds: int = 1,
) -> Tuple[TimeCourseTiffDataset, TimeCourseTiffDataset, TimeCourseTiffDataset]:
    infected_dir = Path(data_cfg["infected_dir"])
    uninfected_dir = Path(data_cfg["uninfected_dir"])
    infected_label = data_cfg.get("infected_label", 1)
    uninfected_label = data_cfg.get("uninfected_label", 0)
    infected_items = _scan_condition(infected_dir, infected_label, "infected")
    uninfected_items = _scan_condition(uninfected_dir, uninfected_label, "uninfected")
    all_items = infected_items + uninfected_items

    ratios = data_cfg.get("split_ratios", [0.7, 0.15, 0.15])
    split_seed = data_cfg.get("split_seed", 42)
    split = _stratified_split(all_items, ratios, seed=split_seed)

    frames_cfg = data_cfg.get("frames")
    train_policy = _resolve_policy(frames_cfg, "train")
    val_policy = _resolve_policy(frames_cfg, "val")
    test_policy = _resolve_policy(frames_cfg, "test")

    if num_folds > 1:
        if fold_index is None:
            raise ValueError("fold_index must be provided when num_folds > 1")
        if fold_index < 0 or fold_index >= num_folds:
            raise ValueError("fold_index out of range")
        cv_items = split.train + split.val
        if len(cv_items) < num_folds:
            raise ValueError("Not enough files to perform the requested number of folds")
        labels = [item.label for item in cv_items]
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=split_seed)
        folds = list(skf.split(range(len(cv_items)), labels))
        train_idx, val_idx = folds[fold_index]
        train_items = [cv_items[i] for i in train_idx]
        val_items = [cv_items[i] for i in val_idx]
    else:
        train_items = split.train
        val_items = split.val

    test_items = split.test

    train_samples = _expand_samples(train_items, train_policy)
    val_samples = _expand_samples(val_items, val_policy)
    test_samples = _expand_samples(test_items, test_policy)

    train_ds = TimeCourseTiffDataset(
        train_samples,
        transform=transforms.get("train"),
        frames_per_hour=train_policy.frames_per_hour,
    )
    val_ds = TimeCourseTiffDataset(
        val_samples,
        transform=transforms.get("val"),
        frames_per_hour=val_policy.frames_per_hour,
    )
    test_ds = TimeCourseTiffDataset(
        test_samples,
        transform=transforms.get("test"),
        frames_per_hour=test_policy.frames_per_hour,
    )
    return train_ds, val_ds, test_ds


def _expand_samples(items: Sequence[DataItem], policy: FrameExtractionPolicy) -> List[FrameSample]:
    samples: List[FrameSample] = []
    frames_per_hour = max(policy.frames_per_hour, 1e-6)
    for item in items:
        if item.condition == "infected":
            frame_indices = policy.infected_indices(item.total_frames)
        else:
            frame_indices = policy.uninfected_indices(item.total_frames)
        for idx in frame_indices:
            hours_since_start = idx / frames_per_hour
            samples.append(
                FrameSample(
                    path=item.path,
                    label=item.label,
                    condition=item.condition,
                    position=item.position,
                    frame_index=idx,
                    total_frames=item.total_frames,
                    hours_since_start=hours_since_start,
                )
            )
    return samples
