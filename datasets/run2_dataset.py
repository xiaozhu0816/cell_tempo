"""
Run2 Dataset for 4-class multi-task learning.

Data layout:
  - 12 wells: a1-a4, b1-b4, c1-c4
  - 36 positions per well (TIFF stacks)
  - 93 frames per TIFF, 30 min interval → 0–46 hours
  - 4 classes: MOI5 (col1), MOI1 (col2), MOI0.1 (col3), Mock (col4)

Split strategy:
  - Row A (a1-a4): split by position → 70% train, 30% test
  - Row B+C (b1-b4, c1-c4): 100% external test
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
from PIL import Image
from torch.utils.data import Dataset

# Pattern matches Run2 filenames like:
#   ..._s5_a1_p01_t01-93_...
_WELL_POS_PATTERN = re.compile(
    r"_s5_(?P<row>[a-c])(?P<col>\d+)_p(?P<pos>\d+)_t(?P<tstart>\d+)-(?P<tend>\d+)_",
    re.IGNORECASE,
)


@dataclass
class Run2Item:
    """A single TIFF stack from Run2."""
    path: Path
    well: str           # e.g. "a1"
    row: str            # e.g. "a"
    col: int            # e.g. 1
    position: str       # e.g. "01"
    condition: str      # e.g. "moi5"
    label: int          # class index 0-3
    total_frames: int   # number of frames in TIFF


@dataclass
class Run2Sample:
    """A single frame sample for training/testing."""
    path: Path
    well: str
    row: str
    position: str
    condition: str
    label: int
    frame_index: int
    total_frames: int
    hours: float        # time in hours from t0


# ---------------------------------------------------------------------------
# Scan and parse Run2 directory
# ---------------------------------------------------------------------------
def _resolve_path(p: str) -> Path:
    """Resolve Linux isilon path to Windows UNC if needed."""
    # Normalise forward slashes for matching
    pn = p.replace("\\", "/")
    # Try Linux → Windows UNC conversion
    prefix = "/isilon/datalake/gurcan_rsch"
    if pn.startswith(prefix):
        remainder = pn[len(prefix):]  # e.g. "/scratch/WSI/..."
        win = r"\\medctr.ad.wfubmc.edu\dfs\gurcan_rsch$" + remainder.replace("/", "\\")
        pp = Path(win)
        if pp.exists():
            return pp
    # Also try if it was already mangled (leading \ on Windows)
    if pn.lstrip("/").startswith("isilon/datalake/gurcan_rsch"):
        remainder = pn.lstrip("/")[len("isilon/datalake/gurcan_rsch"):]
        win = r"\\medctr.ad.wfubmc.edu\dfs\gurcan_rsch$" + remainder.replace("/", "\\")
        pp = Path(win)
        if pp.exists():
            return pp
    pp = Path(p)
    return pp


def scan_run2_dir(data_dir, npy_override_dir=None) -> List[Run2Item]:
    """Scan all TIFFs in the Run2 flat directory.

    If npy_cache/ exists alongside the TIFFs (or npy_override_dir is set),
    uses the .npy files for frame counts (header-only read) and stores the
    .npy path for fast training I/O.  Falls back to tifffile if no cache.

    Args:
        data_dir: path to TIFF directory
        npy_override_dir: optional path to .npy cache on fast local storage
    """
    data_dir = _resolve_path(str(data_dir))
    if not data_dir.exists():
        raise FileNotFoundError(f"Run2 data dir not found: {data_dir}")

    if npy_override_dir:
        npy_dir = Path(npy_override_dir)
    else:
        npy_dir = data_dir / "npy_cache"
    use_npy = npy_dir.is_dir()

    items: List[Run2Item] = []
    for p in sorted(data_dir.glob("*.tif*")):
        m = _WELL_POS_PATTERN.search(p.name)
        if not m:
            continue
        row = m.group("row").lower()
        col = int(m.group("col"))
        well = f"{row}{col}"
        position = m.group("pos")

        npy_path = npy_dir / (p.stem + ".npy") if use_npy else None

        if npy_path and npy_path.exists():
            # Read .npy header only (~128 bytes) to get shape — no data loaded
            with open(npy_path, 'rb') as fh:
                version = np.lib.format.read_magic(fh)
                shape, _, _ = np.lib.format._read_array_header(fh, version)
            total_frames = shape[0]
            item_path = npy_path
        else:
            with tifffile.TiffFile(str(p)) as tif:
                total_frames = len(tif.pages)
            item_path = p

        items.append(Run2Item(
            path=item_path, well=well, row=row, col=col,
            position=position, condition="",  # filled later
            label=-1, total_frames=total_frames,
        ))

    if not items:
        raise RuntimeError(f"No Run2 TIFFs found in {data_dir}")
    if use_npy:
        npy_count = sum(1 for it in items if str(it.path).endswith('.npy'))
        print(f"  [scan] {data_dir.name}: {len(items)} items, "
              f"{npy_count} using npy_cache")
    return items


def assign_labels(
    items: List[Run2Item],
    plate_layout: Dict[str, str],
    class_labels: Dict[str, int],
) -> List[Run2Item]:
    """Assign condition name and class label based on plate layout."""
    for item in items:
        cond = plate_layout.get(item.well)
        if cond is None:
            raise ValueError(f"Well {item.well} not in plate_layout")
        item.condition = cond
        item.label = class_labels[cond]
    return items


# ---------------------------------------------------------------------------
# Split by position within Row A
# ---------------------------------------------------------------------------
def split_row_a(
    items: List[Run2Item],
    train_wells: List[str],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[Run2Item], List[Run2Item]]:
    """
    Split Row A items into train/test by position (within each well).
    Each well has ~36 positions → ~25 train, ~11 test.
    """
    rng = random.Random(seed)
    train_items, test_items = [], []

    for well in sorted(set(train_wells)):
        well_items = [it for it in items if it.well == well]
        if not well_items:
            continue

        # Group by position
        positions = sorted(set(it.position for it in well_items))
        rng.shuffle(positions)

        n_train = max(1, int(len(positions) * train_ratio))
        train_pos = set(positions[:n_train])
        test_pos = set(positions[n_train:])

        for it in well_items:
            if it.position in train_pos:
                train_items.append(it)
            else:
                test_items.append(it)

    return train_items, test_items


# ---------------------------------------------------------------------------
# Expand items → frame samples
# ---------------------------------------------------------------------------
def expand_to_samples(
    items: List[Run2Item],
    frames_per_hour: float = 2.0,
    window_hours: Tuple[float, float] = (0, 46),
    stride: int = 1,
) -> List[Run2Sample]:
    """Expand TIFF items into per-frame samples."""
    samples: List[Run2Sample] = []
    start_frame = max(0, int(math.floor(window_hours[0] * frames_per_hour)))
    end_frame = int(math.floor(window_hours[1] * frames_per_hour))

    for item in items:
        ef = min(end_frame, item.total_frames - 1)
        for fi in range(start_frame, ef + 1, stride):
            hours = fi / frames_per_hour
            samples.append(Run2Sample(
                path=item.path,
                well=item.well,
                row=item.row,
                position=item.position,
                condition=item.condition,
                label=item.label,
                frame_index=fi,
                total_frames=item.total_frames,
                hours=hours,
            ))
    return samples


# ---------------------------------------------------------------------------
# Fast .npy frame reader — direct seek, no mmap
# ---------------------------------------------------------------------------
def _read_npy_frame(npy_path: str, frame_idx: int) -> np.ndarray:
    """Read a single frame from a .npy stack via direct binary seek+read.

    For a stack of shape (N, H, W) dtype uint8, this reads exactly H*W bytes
    at the correct offset. No mmap, no loading the full 450MB file.
    """
    with open(npy_path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        data_offset = f.tell()
        H, W = shape[1], shape[2]
        frame_bytes = H * W * dtype.itemsize
        f.seek(data_offset + frame_idx * frame_bytes)
        frame = np.frombuffer(f.read(frame_bytes), dtype=dtype).reshape(H, W)
    return frame.astype(np.float32)


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------
class Run2Dataset(Dataset):
    """PyTorch Dataset for Run2 frame samples."""

    def __init__(
        self,
        samples: Sequence[Run2Sample],
        transform: Optional[Callable] = None,
        frames_per_hour: float = 2.0,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.frames_per_hour = frames_per_hour

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        frame = self._load_frame(s)
        image = self._to_image(frame)
        if self.transform:
            image = self.transform(image)
        meta = {
            "path": str(s.path),
            "well": s.well,
            "row": s.row,
            "position": s.position,
            "condition": s.condition,
            "frame_index": s.frame_index,
            "hours": float(s.hours),
        }
        return image, s.label, meta

    def _load_frame(self, s: Run2Sample) -> np.ndarray:
        idx = max(0, min(s.total_frames - 1, s.frame_index))
        if str(s.path).endswith('.npy'):
            return _read_npy_frame(str(s.path), idx)
        else:
            with tifffile.TiffFile(str(s.path)) as tif:
                frame = tif.asarray(key=idx)
            return frame.astype(np.float32)

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        """Normalize a float32 frame to [0, 255] uint8."""
        frame = frame - frame.min()
        mx = frame.max()
        if mx > 0:
            frame = frame / mx
        return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    def _to_image(self, frame: np.ndarray) -> Image.Image:
        frame = self._normalize_frame(frame)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        return Image.fromarray(frame)

    def get_metadata(self, idx: int) -> dict:
        """Get metadata without loading the image."""
        s = self.samples[idx]
        return {
            "path": str(s.path),
            "well": s.well,
            "row": s.row,
            "position": s.position,
            "condition": s.condition,
            "label": s.label,
            "frame_index": s.frame_index,
            "hours": float(s.hours),
        }


# ---------------------------------------------------------------------------
# Temporal Dataset — multi-frame input as pseudo-RGB
# ---------------------------------------------------------------------------
class TemporalRun2Dataset(Run2Dataset):
    """
    Dataset returning temporal context: multiple frames stacked as channels.

    For T=3 temporal offsets (e.g., [-6, -3, 0] hours), produces a single
    pseudo-RGB image where:
        R = frame at t + offset[0]  (oldest)
        G = frame at t + offset[1]  (middle)
        B = frame at t + offset[2]  (current)

    This encodes the morphological *trajectory* into a standard 3-channel
    image, requiring zero changes to the ResNet backbone or transforms.

    For frames near the start of the experiment where offsets reach before
    t=0, the earliest available frame (t=0) is used (clamp padding).
    """

    def __init__(
        self,
        samples: Sequence[Run2Sample],
        transform: Optional[Callable] = None,
        frames_per_hour: float = 2.0,
        temporal_offsets_hours: Sequence[float] = (-6, -3, 0),
    ) -> None:
        super().__init__(samples, transform, frames_per_hour)
        self.temporal_offsets_hours = list(temporal_offsets_hours)
        self.offset_frames = [int(h * frames_per_hour) for h in temporal_offsets_hours]
        if len(self.offset_frames) != 3:
            raise ValueError(
                f"temporal_offsets_hours must have exactly 3 entries for "
                f"pseudo-RGB stacking, got {len(self.offset_frames)}: "
                f"{self.temporal_offsets_hours}"
            )

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # Read all needed frames
        is_npy = str(s.path).endswith('.npy')
        channels = []
        for offset in self.offset_frames:
            fi = s.frame_index + offset
            fi = max(0, min(fi, s.total_frames - 1))
            if is_npy:
                raw = _read_npy_frame(str(s.path), fi)
            else:
                with tifffile.TiffFile(str(s.path)) as tif:
                    raw = tif.asarray(key=fi).astype(np.float32)
            channels.append(self._normalize_frame(raw))

        # Stack as pseudo-RGB: [H, W, 3]
        stacked = np.stack(channels, axis=-1)
        pil_img = Image.fromarray(stacked, mode="RGB")

        if self.transform:
            image = self.transform(pil_img)
        else:
            from torchvision import transforms as T
            image = T.ToTensor()(pil_img)

        meta = {
            "path": str(s.path),
            "well": s.well,
            "row": s.row,
            "position": s.position,
            "condition": s.condition,
            "frame_index": s.frame_index,
            "hours": float(s.hours),
        }
        return image, s.label, meta


# ---------------------------------------------------------------------------
# Build function (main entry point)
# ---------------------------------------------------------------------------
def build_run2_datasets(
    data_cfg: Dict,
    transforms_dict: Dict[str, Optional[Callable]],
) -> Dict[str, "Run2Dataset"]:
    """
    Build train + 3 separate test datasets.

    Returns dict with keys:
        "train"  : Row A, 70% positions
        "test_a" : Row A, 30% positions (held-out)
        "test_b" : Row B, all positions (external replicate 1)
        "test_c" : Row C, all positions (external replicate 2)
    """
    # Resolve data directory (Linux ↔ Windows)
    run2_dir_str = data_cfg["run2_dir"]
    run2_dir = _resolve_path(run2_dir_str)
    if not run2_dir.exists():
        raise FileNotFoundError(f"Run2 data dir not found: {run2_dir}")

    plate_layout = data_cfg["plate_layout"]
    class_labels = data_cfg["class_labels"]
    train_wells = data_cfg.get("train_wells", ["a1", "a2", "a3", "a4"])
    train_ratio = data_cfg.get("train_ratio", 0.7)
    split_seed = data_cfg.get("split_seed", 42)

    frames_cfg = data_cfg.get("frames", {})
    fph = frames_cfg.get("frames_per_hour", 2.0)
    window = tuple(frames_cfg.get("window_hours", [0, 46]))
    stride = frames_cfg.get("stride", 1)

    # 1. Scan all TIFFs
    all_items = scan_run2_dir(run2_dir)
    all_items = assign_labels(all_items, plate_layout, class_labels)

    # 2. Separate rows
    row_a_items = [it for it in all_items if it.row == "a"]
    row_b_items = [it for it in all_items if it.row == "b"]
    row_c_items = [it for it in all_items if it.row == "c"]

    # 3. Split Row A by position
    train_items, test_a_items = split_row_a(row_a_items, train_wells, train_ratio, split_seed)

    # 4. Expand to frame samples
    train_samples = expand_to_samples(train_items, fph, window, stride)
    test_a_samples = expand_to_samples(test_a_items, fph, window, stride)
    test_b_samples = expand_to_samples(row_b_items, fph, window, stride)
    test_c_samples = expand_to_samples(row_c_items, fph, window, stride)

    # 5. Build datasets
    datasets = {
        "train": Run2Dataset(train_samples, transform=transforms_dict.get("train"), frames_per_hour=fph),
        "test_a": Run2Dataset(test_a_samples, transform=transforms_dict.get("test"), frames_per_hour=fph),
        "test_b": Run2Dataset(test_b_samples, transform=transforms_dict.get("test"), frames_per_hour=fph),
        "test_c": Run2Dataset(test_c_samples, transform=transforms_dict.get("test"), frames_per_hour=fph),
    }
    return datasets


# ---------------------------------------------------------------------------
# Build function: Train on Row A + Row C, external test on Row B
# ---------------------------------------------------------------------------
def build_run2_trainAC_datasets(
    data_cfg: Dict,
    transforms_dict: Dict[str, Optional[Callable]],
) -> Dict[str, "Run2Dataset"]:
    """
    Build train + 3 separate test datasets for trainAC experiment.

    Split strategy:
        - Row A (a1-a4): split by position → 70% train, 30% test-A
        - Row C (c1-c4): split by position → 70% train, 30% test-C
        - Row B (b1-b4): 100% external test-B
        - Train = Row A 70% + Row C 70% (merged)

    Returns dict with keys:
        "train"  : Row A 70% + Row C 70% (combined)
        "test_a" : Row A 30% (held-out)
        "test_b" : Row B all positions (external replicate)
        "test_c" : Row C 30% (held-out)
    """
    # Resolve data directory (Linux ↔ Windows)
    run2_dir_str = data_cfg["run2_dir"]
    run2_dir = _resolve_path(run2_dir_str)
    if not run2_dir.exists():
        raise FileNotFoundError(f"Run2 data dir not found: {run2_dir}")

    plate_layout = data_cfg["plate_layout"]
    class_labels = data_cfg["class_labels"]
    train_wells_a = data_cfg.get("train_wells_a", ["a1", "a2", "a3", "a4"])
    train_wells_c = data_cfg.get("train_wells_c", ["c1", "c2", "c3", "c4"])
    train_ratio = data_cfg.get("train_ratio", 0.7)
    split_seed = data_cfg.get("split_seed", 42)

    frames_cfg = data_cfg.get("frames", {})
    fph = frames_cfg.get("frames_per_hour", 2.0)
    window = tuple(frames_cfg.get("window_hours", [0, 46]))
    stride = frames_cfg.get("stride", 1)

    # 1. Scan all TIFFs
    all_items = scan_run2_dir(run2_dir)
    all_items = assign_labels(all_items, plate_layout, class_labels)

    # 2. Separate rows
    row_a_items = [it for it in all_items if it.row == "a"]
    row_b_items = [it for it in all_items if it.row == "b"]
    row_c_items = [it for it in all_items if it.row == "c"]

    # 3. Split Row A by position (70/30)
    train_a, test_a_items = split_row_a(row_a_items, train_wells_a,
                                        train_ratio, split_seed)

    # 4. Split Row C by position (70/30) — reuse split_row_a with C wells
    train_c, test_c_items = split_row_a(row_c_items, train_wells_c,
                                        train_ratio, split_seed)

    # 5. Merge train = Row A train + Row C train
    train_items = train_a + train_c

    # 6. Expand to frame samples
    train_samples  = expand_to_samples(train_items, fph, window, stride)
    test_a_samples = expand_to_samples(test_a_items, fph, window, stride)
    test_b_samples = expand_to_samples(row_b_items, fph, window, stride)
    test_c_samples = expand_to_samples(test_c_items, fph, window, stride)

    # 7. Build datasets
    datasets = {
        "train":  Run2Dataset(train_samples, transform=transforms_dict.get("train"), frames_per_hour=fph),
        "test_a": Run2Dataset(test_a_samples, transform=transforms_dict.get("test"), frames_per_hour=fph),
        "test_b": Run2Dataset(test_b_samples, transform=transforms_dict.get("test"), frames_per_hour=fph),
        "test_c": Run2Dataset(test_c_samples, transform=transforms_dict.get("test"), frames_per_hour=fph),
    }
    return datasets


# ---------------------------------------------------------------------------
# Build: Train on single row, external test on the other row (no Row A)
# ---------------------------------------------------------------------------
def build_run2_single_row_datasets(
    data_cfg: Dict,
    transforms_dict: Dict[str, Optional[Callable]],
) -> Dict[str, "Run2Dataset"]:
    """
    Build train + 2 test datasets for single-row experiment.

    Controlled by config keys:
        train_row:    e.g. "c"  → Row C wells for 70/30 split
        external_row: e.g. "b"  → Row B wells for 100% external test

    Returns dict with keys:
        "train"         : train_row 70% positions
        "test_held_out" : train_row 30% positions (held-out)
        "test_external" : external_row all positions (external replicate)
    """
    run2_dir = _resolve_path(str(data_cfg["run2_dir"]))
    if not run2_dir.exists():
        raise FileNotFoundError(f"Run2 data dir not found: {run2_dir}")

    plate_layout = data_cfg["plate_layout"]
    class_labels = data_cfg["class_labels"]
    train_row = data_cfg["train_row"].lower()        # e.g. "c"
    external_row = data_cfg["external_row"].lower()   # e.g. "b"
    train_wells = [f"{train_row}{c}" for c in range(1, 5)]       # c1-c4
    external_wells = [f"{external_row}{c}" for c in range(1, 5)] # b1-b4
    train_ratio = data_cfg.get("train_ratio", 0.7)
    split_seed = data_cfg.get("split_seed", 42)

    frames_cfg = data_cfg.get("frames", {})
    fph = frames_cfg.get("frames_per_hour", 2.0)
    window = tuple(frames_cfg.get("window_hours", [0, 46]))
    stride = frames_cfg.get("stride", 1)

    # 1. Scan all TIFFs
    all_items = scan_run2_dir(run2_dir)
    all_items = assign_labels(all_items, plate_layout, class_labels)

    # 2. Separate rows (exclude Row A entirely)
    train_row_items = [it for it in all_items if it.row == train_row]
    ext_row_items   = [it for it in all_items if it.row == external_row]

    # 3. Split train row by position (70/30)
    train_items, test_held_items = split_row_a(
        train_row_items, train_wells, train_ratio, split_seed)

    # 4. Expand to frame samples
    train_samples    = expand_to_samples(train_items, fph, window, stride)
    test_held_samples = expand_to_samples(test_held_items, fph, window, stride)
    test_ext_samples = expand_to_samples(ext_row_items, fph, window, stride)

    # 5. Build datasets
    datasets = {
        "train":         Run2Dataset(train_samples, transform=transforms_dict.get("train"), frames_per_hour=fph),
        "test_held_out": Run2Dataset(test_held_samples, transform=transforms_dict.get("test"), frames_per_hour=fph),
        "test_external": Run2Dataset(test_ext_samples, transform=transforms_dict.get("test"), frames_per_hour=fph),
    }
    return datasets


# ---------------------------------------------------------------------------
# Build: Cross-dataset — one dataset train/test, another dataset external
# ---------------------------------------------------------------------------
def _split_all_wells(
    items: List[Run2Item],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[Run2Item], List[Run2Item]]:
    """Split ALL wells by position (70/30) — used for cross-dataset setup."""
    rng = random.Random(seed)
    train_items, test_items = [], []

    wells = sorted(set(it.well for it in items))
    for well in wells:
        well_items = [it for it in items if it.well == well]
        positions = sorted(set(it.position for it in well_items))
        rng.shuffle(positions)
        n_train = max(1, int(len(positions) * train_ratio))
        train_pos = set(positions[:n_train])

        for it in well_items:
            if it.position in train_pos:
                train_items.append(it)
            else:
                test_items.append(it)

    return train_items, test_items


def build_cross_dataset(
    data_cfg: Dict,
    transforms_dict: Dict[str, Optional[Callable]],
) -> Dict[str, "Run2Dataset"]:
    """
    Build cross-dataset experiment.

    Config must provide:
        train_dir:    path to training dataset (all 12 wells → 70/30 split)
        external_dir: path to external dataset (all 12 wells → 100% test)

    Returns dict with keys:
        "train"         : 70% positions from train_dir
        "test_internal" : 30% positions from train_dir
        "test_external" : all positions from external_dir
    """
    train_dir = _resolve_path(str(data_cfg["train_dir"]))
    ext_dir   = _resolve_path(str(data_cfg["external_dir"]))
    if not train_dir.exists():
        raise FileNotFoundError(f"Train data dir not found: {train_dir}")
    if not ext_dir.exists():
        raise FileNotFoundError(f"External data dir not found: {ext_dir}")

    plate_layout = data_cfg["plate_layout"]
    class_labels = data_cfg["class_labels"]
    train_ratio  = data_cfg.get("train_ratio", 0.7)
    split_seed   = data_cfg.get("split_seed", 42)

    frames_cfg = data_cfg.get("frames", {})
    fph    = frames_cfg.get("frames_per_hour", 2.0)
    window = tuple(frames_cfg.get("window_hours", [0, 47.5]))
    stride = frames_cfg.get("stride", 1)

    # 1. Scan both directories
    train_all   = scan_run2_dir(train_dir)
    ext_all     = scan_run2_dir(ext_dir)
    train_all   = assign_labels(train_all, plate_layout, class_labels)
    ext_all     = assign_labels(ext_all, plate_layout, class_labels)

    # 2. Split training dataset by position (70/30) across all wells
    train_items, test_int_items = _split_all_wells(
        train_all, train_ratio, split_seed)

    # 3. Expand to frame-level samples
    train_samples    = expand_to_samples(train_items, fph, window, stride)
    test_int_samples = expand_to_samples(test_int_items, fph, window, stride)
    ext_samples      = expand_to_samples(ext_all, fph, window, stride)

    # 4. Build datasets — choose class based on temporal config
    temporal_cfg = data_cfg.get("temporal", None)
    if temporal_cfg and temporal_cfg.get("enabled", False):
        offsets = tuple(temporal_cfg.get("offsets_hours", [-6, -3, 0]))
        def _make_ds(samples, tx):
            return TemporalRun2Dataset(samples, transform=tx,
                                       frames_per_hour=fph,
                                       temporal_offsets_hours=offsets)
    else:
        def _make_ds(samples, tx):
            return Run2Dataset(samples, transform=tx, frames_per_hour=fph)

    datasets = {
        "train":         _make_ds(train_samples,    transforms_dict.get("train")),
        "test_internal": _make_ds(test_int_samples, transforms_dict.get("test")),
        "test_external": _make_ds(ext_samples,      transforms_dict.get("test")),
    }
    return datasets


def build_multi_train_external_dataset(
    data_cfg: Dict,
    transforms_dict: Dict[str, Optional[Callable]],
) -> Dict[str, "Run2Dataset"]:
    """
    Build cross-dataset experiment with multiple training datasets.

    Config must provide:
        train_dirs:    list of dataset paths (each split 70/30 internally)
        external_dir:  path to external dataset (100% test)

    Returns dict with keys:
        "train"         : merged 70% from each train dataset
        "test_internal" : merged 30% from each train dataset
        "test_external" : all positions from external dataset
    """
    train_dirs_raw = data_cfg.get("train_dirs", [])
    if not train_dirs_raw:
        raise ValueError("data.train_dirs is required and must be non-empty")

    train_dirs = [_resolve_path(str(p)) for p in train_dirs_raw]
    ext_dir = _resolve_path(str(data_cfg["external_dir"]))

    for td in train_dirs:
        if not td.exists():
            raise FileNotFoundError(f"Train data dir not found: {td}")
    if not ext_dir.exists():
        raise FileNotFoundError(f"External data dir not found: {ext_dir}")

    plate_layout = data_cfg["plate_layout"]
    class_labels = data_cfg["class_labels"]
    train_ratio = data_cfg.get("train_ratio", 0.7)
    split_seed = data_cfg.get("split_seed", 42)

    frames_cfg = data_cfg.get("frames", {})
    fph = frames_cfg.get("frames_per_hour", 2.0)
    window = tuple(frames_cfg.get("window_hours", [0, 47.5]))
    stride = frames_cfg.get("stride", 1)

    merged_train_items: List[Run2Item] = []
    merged_test_int_items: List[Run2Item] = []

    for i, td in enumerate(train_dirs):
        all_items = scan_run2_dir(td)
        all_items = assign_labels(all_items, plate_layout, class_labels)
        train_items, test_int_items = _split_all_wells(
            all_items, train_ratio, split_seed + i
        )
        merged_train_items.extend(train_items)
        merged_test_int_items.extend(test_int_items)

    ext_all = scan_run2_dir(ext_dir)
    ext_all = assign_labels(ext_all, plate_layout, class_labels)

    train_samples = expand_to_samples(merged_train_items, fph, window, stride)
    test_int_samples = expand_to_samples(merged_test_int_items, fph, window, stride)
    ext_samples = expand_to_samples(ext_all, fph, window, stride)

    # Choose dataset class based on temporal config
    temporal_cfg = data_cfg.get("temporal", None)
    if temporal_cfg and temporal_cfg.get("enabled", False):
        offsets = tuple(temporal_cfg.get("offsets_hours", [-6, -3, 0]))
        def _make_ds(samples, tx):
            return TemporalRun2Dataset(samples, transform=tx,
                                       frames_per_hour=fph,
                                       temporal_offsets_hours=offsets)
    else:
        def _make_ds(samples, tx):
            return Run2Dataset(samples, transform=tx, frames_per_hour=fph)

    datasets = {
        "train":         _make_ds(train_samples,    transforms_dict.get("train")),
        "test_internal": _make_ds(test_int_samples, transforms_dict.get("test")),
        "test_external": _make_ds(ext_samples,      transforms_dict.get("test")),
    }
    return datasets


# ---------------------------------------------------------------------------
# Build: Row-based split with per-directory plate layouts
# ---------------------------------------------------------------------------
def build_row_split_dataset(
    data_cfg: Dict,
    transforms_dict: Dict[str, Optional[Callable]],
) -> Dict[str, "Run2Dataset"]:
    """
    Row-based train/test split across multiple runs, each with its own plate layout.

    Config format:
      data:
        datasets:
          - dir: /path/to/run22
            plate_layout: {a1: moi5, a2: moi1, a3: moi01, a4: mock, ...}
            well_overrides: {c3: train, a1: test}  # optional per-well override
          - dir: /path/to/run4
            plate_layout: {a1: moi01, a2: moi5, a3: moi1, a4: mock, ...}
        class_labels: {moi5: 0, moi1: 1, moi01: 2, mock: 3}
        train_rows: [a, b]
        test_rows: [c]

    Returns dict with keys:
        "train"         : train_rows from all datasets (+ override wells)
        "test_internal" : test_rows from all datasets (+ override wells)
    """
    ds_cfgs = data_cfg.get("datasets", [])
    if not ds_cfgs:
        raise ValueError("data.datasets is required and must be non-empty")

    class_labels = data_cfg["class_labels"]
    train_rows = [r.lower() for r in data_cfg.get("train_rows", ["a", "b"])]
    test_rows  = [r.lower() for r in data_cfg.get("test_rows", ["c"])]

    frames_cfg = data_cfg.get("frames", {})
    fph    = frames_cfg.get("frames_per_hour", 2.0)
    window = tuple(frames_cfg.get("window_hours", [0, 47.5]))
    stride = frames_cfg.get("stride", 1)

    all_train_items: List[Run2Item] = []
    all_test_items: List[Run2Item] = []

    for idx, ds_entry in enumerate(ds_cfgs):
        d = _resolve_path(str(ds_entry["dir"]))
        if not d.exists():
            raise FileNotFoundError(f"Data dir not found: {d}")
        plate_layout = ds_entry["plate_layout"]
        local_npy = ds_entry.get("npy_dir", None)

        # Run tag for distinguishing wells across runs (e.g. "R2_c1", "R3_c1")
        run_tag = ds_entry.get("name", f"R{idx+1}")

        items = scan_run2_dir(d, npy_override_dir=local_npy)
        items = assign_labels(items, plate_layout, class_labels)

        # Prefix well names so wells from different runs are distinguishable
        for it in items:
            it.well = f"{run_tag}_{it.well}"

        # well_overrides: move specific wells between train/test
        # Format: {well_name: "train"/"test"} e.g. {"c3": "train", "a1": "test"}
        well_overrides = ds_entry.get("well_overrides", {})

        for it in items:
            well_key = f"{it.row}{it.col}"  # e.g. "c3", "a1"
            override = well_overrides.get(well_key, None)
            if override == "train":
                all_train_items.append(it)
            elif override == "test":
                all_test_items.append(it)
            elif it.row in train_rows:
                all_train_items.append(it)
            elif it.row in test_rows:
                all_test_items.append(it)

    # Test can use a different (wider) window than train
    # e.g. train on [3, 47.5]h but test on full [0, 47.5]h
    test_window = tuple(frames_cfg.get("test_window_hours", window))

    train_samples = expand_to_samples(all_train_items, fph, window, stride)
    test_samples  = expand_to_samples(all_test_items, fph, test_window, stride)

    # Choose dataset class based on temporal config
    temporal_cfg = data_cfg.get("temporal", None)
    if temporal_cfg and temporal_cfg.get("enabled", False):
        offsets = tuple(temporal_cfg.get("offsets_hours", [-6, -3, 0]))
        def _make_ds(samples, tx):
            return TemporalRun2Dataset(samples, transform=tx,
                                       frames_per_hour=fph,
                                       temporal_offsets_hours=offsets)
        def _make_ds_singleframe(samples, tx):
            return TemporalRun2Dataset(samples, transform=tx,
                                       frames_per_hour=fph,
                                       temporal_offsets_hours=(0, 0, 0))
    else:
        def _make_ds(samples, tx):
            return Run2Dataset(samples, transform=tx, frames_per_hour=fph)
        _make_ds_singleframe = None

    datasets = {
        "train":         _make_ds(train_samples, transforms_dict.get("train")),
        "test_internal": _make_ds(test_samples,  transforms_dict.get("test")),
    }

    # Ablation: temporal model tested with single-frame input (R=G=B=t)
    if _make_ds_singleframe is not None:
        datasets["test_internal_singleframe"] = _make_ds_singleframe(
            test_samples, transforms_dict.get("test"))

    return datasets
