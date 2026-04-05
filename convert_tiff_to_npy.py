"""
Convert TIFF stacks to .npy files for fast training I/O.

Each TIFF stack (e.g., 96 frames of 2048×2448 uint8) is saved as a single
.npy file with shape [N_frames, H, W]. During training, np.load(mmap_mode='r')
lets us read any frame with a simple array index — no TIFF header parsing,
no decompression, no tifffile overhead.

Usage:
    python convert_tiff_to_npy.py

Output structure (alongside original TIFFs):
    <run_dir>/npy_cache/<original_name>.npy
"""
import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile
from tqdm import tqdm


def convert_one(tiff_path: str, out_path: str) -> str:
    """Convert a single TIFF stack to .npy."""
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            stack = tif.asarray()  # [N, H, W] uint8
        np.save(out_path, stack)
        return f"OK: {Path(tiff_path).name} -> {stack.shape}"
    except Exception as e:
        return f"FAIL: {Path(tiff_path).name} -> {e}"


def convert_directory(data_dir: str, max_workers: int = 8):
    """Convert all TIFFs in a directory to .npy in a subdirectory."""
    data_dir = Path(data_dir)
    out_dir = data_dir / "npy_cache"
    out_dir.mkdir(exist_ok=True)

    tiffs = sorted([f for f in data_dir.iterdir()
                    if f.suffix.lower() in ('.tif', '.tiff')])
    print(f"\n{'='*60}")
    print(f"Directory: {data_dir}")
    print(f"Output:    {out_dir}")
    print(f"TIFFs:     {len(tiffs)}")
    print(f"Workers:   {max_workers}")
    print(f"{'='*60}")

    # Skip already converted
    todo = []
    for t in tiffs:
        npy_path = out_dir / (t.stem + ".npy")
        if npy_path.exists():
            continue
        todo.append((str(t), str(npy_path)))

    if not todo:
        print("All files already converted, skipping.")
        return

    print(f"To convert: {len(todo)} (skipping {len(tiffs) - len(todo)} existing)")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(convert_one, t, o): t for t, o in todo}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            result = f.result()
            if result.startswith("FAIL"):
                print(f"  {result}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of parallel workers")
    args = ap.parse_args()

    data_dirs = [
        "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/DATA/GMU_cell_1023/HBMVEC/Validation_Run2_3-11-26/Enhanced contour",
        "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/DATA/GMU_cell_1023/HBMVEC/Validation_Run3_3-13-26",
        "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/DATA/GMU_cell_1023/HBMVEC/Validation_Run_4_3-15-26/Enhanced contour",
    ]

    for d in data_dirs:
        convert_directory(d, max_workers=args.workers)

    print("\nDone! All TIFF stacks converted to .npy")
    print("The dataset code will auto-detect npy_cache/ and use it.")


if __name__ == "__main__":
    main()
