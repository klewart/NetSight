#!/usr/bin/env python3
"""
download_dataset.py — Automated Dataset Download & Preparation
===============================================================
Downloads and organizes road segmentation datasets for NetSight training:

1. DeepGlobe Road Extraction (Kaggle) — primary training data
2. Sentinel-2 India Roads (Kaggle) — India-specific fine-tuning
3. Custom OSM + ESRI generation (built-in data_generator.py)

Usage:
    # Download DeepGlobe from Kaggle (requires kaggle CLI)
    python download_dataset.py --source deepglobe --output_dir training_data

    # Download Sentinel-2 India Roads
    python download_dataset.py --source sentinel2_india --output_dir training_data

    # Validate and standardize an existing dataset
    python download_dataset.py --source local --input_dir my_data --output_dir training_data

    # Generate OSM data for Indian cities
    python download_dataset.py --source osm --output_dir training_data

Author: NetSight Project (ISRO NNRMS / Disaster Management Framework)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("netsight.download")


# ── Dataset Sources ──────────────────────────────────────────────────────

DATASETS = {
    "deepglobe": {
        "name": "DeepGlobe Road Extraction Dataset",
        "kaggle_slug": "balraj98/deepglobe-road-extraction-dataset",
        "description": "6,226 RGB satellite images (1024×1024) with binary road masks",
        "resolution": "50cm/px",
        "img_suffix": "_sat.jpg",
        "mask_suffix": "_mask.png",
    },
    "sentinel2_india": {
        "name": "Sentinel-2 India Roads Dataset",
        "kaggle_slug": "sagar100/sentinel-2-road-detection-india",
        "description": "5,634 samples (256×256) covering Indian urban/rural/mountain terrain",
        "resolution": "10m/px",
        "img_suffix": ".png",
        "mask_suffix": ".png",
    },
}


def check_kaggle_cli() -> bool:
    """Check if the Kaggle CLI is installed and configured."""
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            log.info(f"Kaggle CLI found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    log.error(
        "Kaggle CLI not found. Install with:\n"
        "  pip install kaggle\n"
        "Then configure:\n"
        "  1. Go to https://www.kaggle.com/settings → Create New API Token\n"
        "  2. Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\\Users\\<you>\\.kaggle\\ (Windows)"
    )
    return False


def download_kaggle_dataset(slug: str, output_dir: Path) -> bool:
    """Download a dataset from Kaggle using the CLI."""
    if not check_kaggle_cli():
        return False

    log.info(f"Downloading {slug} from Kaggle...")
    raw_dir = output_dir / "raw_download"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir), "--unzip"],
            check=True,
        )
        log.info(f"Downloaded and extracted to {raw_dir}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Kaggle download failed: {e}")
        return False


def organize_deepglobe(raw_dir: Path, output_dir: Path, tile_size: int = 512):
    """
    Organize DeepGlobe dataset into standardized structure.

    DeepGlobe naming: {id}_sat.jpg (image), {id}_mask.png (mask)
    Output: images/{id}.png, masks/{id}.png (resized to tile_size)
    """
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Find all satellite images
    sat_files = list(raw_dir.rglob("*_sat.jpg")) + list(raw_dir.rglob("*_sat.png"))
    log.info(f"Found {len(sat_files)} DeepGlobe satellite images")

    processed = 0
    skipped = 0

    for sat_path in sat_files:
        stem = sat_path.stem.replace("_sat", "")
        mask_path = sat_path.parent / f"{stem}_mask.png"

        if not mask_path.exists():
            log.warning(f"No mask for {sat_path.name}, skipping")
            skipped += 1
            continue

        try:
            # Load and resize image
            img = Image.open(sat_path).convert("RGB")
            img = img.resize((tile_size, tile_size), Image.LANCZOS)
            img.save(img_dir / f"{stem}.png")

            # Load, binarize, and resize mask
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((tile_size, tile_size), Image.NEAREST)  # NEAREST for masks!
            mask_arr = np.array(mask)
            mask_arr = ((mask_arr > 127) * 255).astype(np.uint8)
            Image.fromarray(mask_arr).save(mask_dir / f"{stem}.png")

            processed += 1
        except Exception as e:
            log.warning(f"Failed to process {sat_path.name}: {e}")
            skipped += 1

    log.info(f"DeepGlobe: {processed} pairs organized, {skipped} skipped")
    return processed


def organize_sentinel2_india(raw_dir: Path, output_dir: Path, tile_size: int = 512):
    """
    Organize Sentinel-2 India Roads dataset.

    This dataset typically has images/ and masks/ subdirectories.
    Images are 256×256 — we resize to tile_size (512) for consistency.
    """
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Try to find the image/mask directories in the download
    src_img_dirs = list(raw_dir.rglob("images_png")) + list(raw_dir.rglob("images"))
    src_mask_dirs = list(raw_dir.rglob("masks_png")) + list(raw_dir.rglob("masks"))

    if not src_img_dirs:
        # Fallback: look for any PNG files
        all_pngs = list(raw_dir.rglob("*.png"))
        log.info(f"No standard directory structure found. Found {len(all_pngs)} PNGs total.")
        return 0

    src_img_dir = src_img_dirs[0]
    src_mask_dir = src_mask_dirs[0] if src_mask_dirs else None

    if src_mask_dir is None:
        log.error("No mask directory found in Sentinel-2 download")
        return 0

    img_files = sorted(src_img_dir.glob("*.png")) + sorted(src_img_dir.glob("*.tif"))
    log.info(f"Found {len(img_files)} Sentinel-2 images in {src_img_dir}")

    processed = 0
    for img_path in img_files:
        mask_path = src_mask_dir / img_path.name
        if not mask_path.exists():
            # Try alternate extensions
            for ext in [".png", ".tif", ".jpg"]:
                alt = src_mask_dir / f"{img_path.stem}{ext}"
                if alt.exists():
                    mask_path = alt
                    break
            else:
                continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((tile_size, tile_size), Image.LANCZOS)
            img.save(img_dir / f"s2_{img_path.stem}.png")

            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((tile_size, tile_size), Image.NEAREST)
            mask_arr = np.array(mask)
            mask_arr = ((mask_arr > 127) * 255).astype(np.uint8)
            Image.fromarray(mask_arr).save(mask_dir / f"s2_{img_path.stem}.png")

            processed += 1
        except Exception as e:
            log.warning(f"Failed: {img_path.name}: {e}")

    log.info(f"Sentinel-2 India: {processed} pairs organized")
    return processed


def validate_dataset(data_dir: Path) -> Tuple[int, int, int]:
    """
    Validate dataset integrity. Returns (valid, corrupt, empty).

    Checks:
    - Image/mask pairs exist
    - Images are RGB and loadable
    - Masks are grayscale and binary
    - No completely black images (sensor failure)
    - No completely white masks (labeling error)
    """
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        log.error(f"Dataset directories not found in {data_dir}")
        return 0, 0, 0

    img_files = set(p.stem for p in img_dir.glob("*.png"))
    mask_files = set(p.stem for p in mask_dir.glob("*.png"))
    paired = sorted(img_files & mask_files)

    valid, corrupt, empty = 0, 0, 0

    for stem in paired:
        try:
            img = Image.open(img_dir / f"{stem}.png").convert("RGB")
            mask = Image.open(mask_dir / f"{stem}.png").convert("L")

            img_arr = np.array(img)
            mask_arr = np.array(mask)

            # Check for completely black image (sensor failure)
            if img_arr.mean() < 5:
                log.debug(f"Removing nearly-black image: {stem}")
                os.remove(img_dir / f"{stem}.png")
                os.remove(mask_dir / f"{stem}.png")
                corrupt += 1
                continue

            # Check for completely white mask (labeling error)
            if mask_arr.mean() > 250:
                log.debug(f"Removing over-saturated mask: {stem}")
                os.remove(img_dir / f"{stem}.png")
                os.remove(mask_dir / f"{stem}.png")
                corrupt += 1
                continue

            # Check for completely empty mask (no roads at all)
            if mask_arr.max() == 0:
                empty += 1
                # Keep these — they teach the model about non-road areas

            valid += 1
        except Exception as e:
            log.warning(f"Corrupt pair {stem}: {e}")
            corrupt += 1

    log.info(f"Validation: {valid} valid, {corrupt} corrupt (removed), {empty} empty masks")
    return valid, corrupt, empty


def generate_splits(data_dir: Path, val_ratio: float = 0.15, test_ratio: float = 0.05, seed: int = 42):
    """Generate train/val/test split files."""
    img_dir = data_dir / "images"
    stems = sorted(p.stem for p in img_dir.glob("*.png"))

    np.random.seed(seed)
    np.random.shuffle(stems)

    n = len(stems)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    train = stems[:n_train]
    val = stems[n_train:n_train + n_val]
    test = stems[n_train + n_val:]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        split_file = data_dir / f"{name}.txt"
        with open(split_file, "w") as f:
            f.write("\n".join(split))
        log.info(f"  {name}: {len(split)} samples → {split_file}")


def main():
    parser = argparse.ArgumentParser(description="NetSight Dataset Download & Preparation")
    parser.add_argument("--source", type=str, required=True,
                        choices=["deepglobe", "sentinel2_india", "osm", "local"],
                        help="Dataset source to download")
    parser.add_argument("--output_dir", type=str, default="training_data",
                        help="Output directory for organized dataset")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory for local dataset (--source local)")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Resize all images/masks to this size")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Validate dataset after processing")
    parser.add_argument("--splits", action="store_true", default=True,
                        help="Generate train/val/test splits")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NetSight — Dataset Download & Preparation Pipeline")
    print("=" * 60)

    if args.source == "deepglobe":
        info = DATASETS["deepglobe"]
        print(f"\n  Dataset: {info['name']}")
        print(f"  Size: {info['description']}")
        print(f"  Resolution: {info['resolution']}")
        print()

        raw_dir = output_dir / "raw_download"
        has_local_data = list(raw_dir.rglob("*_sat.jpg")) or list(raw_dir.rglob("*_sat.png"))
        
        if has_local_data:
            print("  Found existing manual download. Proceeding to organize...")
            organize_deepglobe(raw_dir, output_dir, args.tile_size)
        elif download_kaggle_dataset(info["kaggle_slug"], output_dir):
            organize_deepglobe(raw_dir, output_dir, args.tile_size)
        else:
            print("\n  MANUAL DOWNLOAD INSTRUCTIONS:")
            print("  1. Go to: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset")
            print("  2. Click 'Download' and extract the ZIP")
            print(f"  3. Place the extracted folder in: {output_dir / 'raw_download'}")
            print(f"  4. Re-run: python download_dataset.py --source deepglobe --output_dir {args.output_dir}")

    elif args.source == "sentinel2_india":
        info = DATASETS["sentinel2_india"]
        print(f"\n  Dataset: {info['name']}")
        print(f"  Size: {info['description']}")
        print()

        if download_kaggle_dataset(info["kaggle_slug"], output_dir):
            organize_sentinel2_india(output_dir / "raw_download", output_dir, args.tile_size)
        else:
            print("\n  MANUAL DOWNLOAD INSTRUCTIONS:")
            print("  1. Search Kaggle for 'Sentinel-2 Road Detection India'")
            print("  2. Download and extract the dataset")
            print(f"  3. Place in: {output_dir / 'raw_download'}")
            print(f"  4. Re-run this script")

    elif args.source == "osm":
        print("\n  Generating OSM + ESRI satellite data for Indian cities...")
        # Import and run the data generator
        sys.path.insert(0, str(Path(__file__).parent))
        from my_agents.src.data_generator import main as gen_main
        gen_main()
        # Copy generated data to output
        src = Path("data/india_dataset")
        if src.exists():
            for subdir in ["images", "masks"]:
                src_sub = src / subdir
                dst_sub = output_dir / subdir
                dst_sub.mkdir(parents=True, exist_ok=True)
                for f in src_sub.glob("*.png"):
                    shutil.copy2(f, dst_sub / f.name)
            log.info(f"Copied OSM data to {output_dir}")

    elif args.source == "local":
        if args.input_dir is None:
            log.error("--input_dir is required for --source local")
            return
        input_dir = Path(args.input_dir)
        log.info(f"Organizing local dataset from {input_dir}")
        # Just validate existing structure
        for subdir in ["images", "masks"]:
            src = input_dir / subdir
            dst = output_dir / subdir
            if src.exists() and src != dst:
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.glob("*"):
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                        shutil.copy2(f, dst / f.name)

    # Validate
    if args.validate:
        print("\n  Validating dataset...")
        validate_dataset(output_dir)

    # Generate splits
    if args.splits:
        print("\n  Generating train/val/test splits...")
        generate_splits(output_dir)

    print("\n" + "=" * 60)
    print(f"  Dataset ready at: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
