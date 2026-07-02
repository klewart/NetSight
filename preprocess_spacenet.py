#!/usr/bin/env python3
"""
preprocess_spacenet.py
======================
SpaceNet 3 Road-Network Pre-processing Pipeline

Reads 3-band GeoTIFF satellite images and matching GeoJSON road-network
vector files, rasterizes road LineStrings into binary masks with a
configurable buffer width, and writes uniform 512×512 PNG image/mask
pairs ready for a PyTorch DataLoader.

Dependencies:
    pip install rasterio geopandas shapely numpy Pillow

Usage:
    python preprocess_spacenet.py                          # uses defaults
    python preprocess_spacenet.py --img_dir  data/images   \
                                  --vec_dir  data/geojson  \
                                  --out_dir  data/processed \
                                  --tile_size 512           \
                                  --buffer_px 4
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from PIL import Image
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Common SpaceNet image-name patterns:
#   SN3_roads_train_AOI_2_Vegas_PS-RGB_img1.tif
#   AOI_2_Vegas_Scene_001.tif
# We extract the "image ID" as the stem minus any known prefix/suffix tags.
_STRIP_PREFIXES = re.compile(
    r"^(?:SN3_roads_(?:train|test)_)?",  # optional SN3 prefix
    re.IGNORECASE,
)
_STRIP_SUFFIXES = re.compile(
    r"(?:_PS-(?:RGB|MS))?(?:_img\d+)?$",  # optional band / img suffix
    re.IGNORECASE,
)


def _extract_image_id(stem: str) -> str:
    """
    Derive a normalised Image ID from a filename stem so that an image
    and its GeoJSON can be matched regardless of minor naming differences.

    Examples
    --------
    >>> _extract_image_id("SN3_roads_train_AOI_2_Vegas_PS-RGB_img1")
    'aoi_2_vegas'
    >>> _extract_image_id("AOI_2_Vegas_Scene_001")
    'aoi_2_vegas_scene_001'
    """
    s = _STRIP_PREFIXES.sub("", stem)
    s = _STRIP_SUFFIXES.sub("", s)
    return s.strip("_").lower()


def _build_index(
    directory: Path, extensions: Tuple[str, ...]
) -> Dict[str, Path]:
    """
    Walk *directory* and return ``{normalised_image_id: filepath}`` for
    every file whose suffix is in *extensions*.
    """
    index: Dict[str, Path] = {}
    if not directory.is_dir():
        logger.error("Directory does not exist: %s", directory)
        return index

    for fpath in sorted(directory.rglob("*")):
        if fpath.suffix.lower() in extensions:
            key = _extract_image_id(fpath.stem)
            if key in index:
                logger.warning(
                    "Duplicate ID '%s' – keeping %s, ignoring %s",
                    key,
                    index[key].name,
                    fpath.name,
                )
            else:
                index[key] = fpath
    return index


def _read_image_as_array(
    img_path: Path, tile_size: int
) -> Optional[np.ndarray]:
    """
    Read a raster image (GeoTIFF or plain TIFF) and return a
    (tile_size, tile_size, 3) uint8 numpy array.

    Handles both georeferenced (rasterio) and non-georeferenced (PIL)
    images, resizing to tile_size × tile_size if necessary.
    """
    try:
        with rasterio.open(img_path) as src:
            # Read bands 1-3 (RGB). Some SpaceNet images have >3 bands.
            bands_to_read = min(src.count, 3)
            data = src.read(list(range(1, bands_to_read + 1)))  # (C, H, W)

            # If fewer than 3 bands, duplicate to fill RGB
            if bands_to_read < 3:
                data = np.concatenate(
                    [data] * (3 // bands_to_read + 1), axis=0
                )[:3]

            img_arr = np.transpose(data, (1, 2, 0))  # (H, W, C)
    except rasterio.errors.RasterioIOError:
        logger.warning(
            "rasterio could not open %s – falling back to PIL", img_path.name
        )
        try:
            img_arr = np.array(Image.open(img_path).convert("RGB"))
        except Exception as exc:
            logger.error("Failed to read image %s: %s", img_path.name, exc)
            return None

    # Resize to tile_size × tile_size
    pil_img = Image.fromarray(img_arr.astype(np.uint8)).resize(
        (tile_size, tile_size), Image.LANCZOS
    )
    return np.array(pil_img)


def _get_raster_meta(img_path: Path) -> Tuple[Optional[rasterio.Affine], Optional[object], int, int]:
    """
    Return (transform, crs, width, height) from a raster file.
    Returns (None, None, w, h) if the file has no georeference.
    """
    try:
        with rasterio.open(img_path) as src:
            transform = src.transform
            crs = src.crs
            w, h = src.width, src.height

            # Check for an identity / missing transform
            identity = rasterio.transform.Affine.identity()
            if transform is None or transform == identity:
                return None, None, w, h
            return transform, crs, w, h
    except Exception:
        return None, None, 0, 0


def _rasterize_roads(
    geojson_path: Path,
    img_path: Path,
    tile_size: int,
    buffer_px: int,
) -> Optional[np.ndarray]:
    """
    Rasterize road LineStrings from a GeoJSON file into a binary mask.

    The buffer is applied in **pixel space** (after projecting geometries
    to image coordinates) so that ``buffer_px`` always represents the
    number of pixels of road half-width regardless of the image GSD.

    Returns
    -------
    np.ndarray  (tile_size, tile_size) with dtype uint8, values {0, 255}.
    None on failure.
    """
    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as exc:
        logger.error("Failed to read GeoJSON %s: %s", geojson_path.name, exc)
        return None

    if gdf.empty:
        logger.warning(
            "GeoJSON %s has no features – returning blank mask",
            geojson_path.name,
        )
        return np.zeros((tile_size, tile_size), dtype=np.uint8)

    # Keep only line geometries
    gdf = gdf[
        gdf.geometry.apply(
            lambda g: isinstance(g, (LineString, MultiLineString)) if g else False
        )
    ]
    if gdf.empty:
        logger.warning(
            "No LineString geometries in %s – returning blank mask",
            geojson_path.name,
        )
        return np.zeros((tile_size, tile_size), dtype=np.uint8)

    # ---- Determine coordinate mapping --------------------------------
    transform, crs, src_w, src_h = _get_raster_meta(img_path)

    if transform is not None and crs is not None:
        # ---- Georeferenced image: project GeoJSON CRS → image CRS ----
        if gdf.crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)

        # Transform geographic coords → pixel coords using the inverse
        # of the raster's affine transform.
        inv_transform = ~transform

        def _to_pixel(geom):
            if geom.is_empty:
                return geom
            coords = list(geom.coords) if isinstance(geom, LineString) else [
                c for part in geom.geoms for c in part.coords
            ]
            pixel_coords = [inv_transform * (x, y) for x, y in coords]
            if isinstance(geom, LineString):
                return LineString(pixel_coords) if len(pixel_coords) >= 2 else geom
            return geom  # fallback

        gdf["geometry"] = gdf.geometry.apply(_to_pixel)
    # else: coordinates are already in pixel space (synthetic data)

    # ---- Scale pixel coords to target tile_size ----------------------
    if src_w > 0 and src_h > 0 and (src_w != tile_size or src_h != tile_size):
        sx = tile_size / src_w
        sy = tile_size / src_h

        from shapely.affinity import scale as shapely_scale

        gdf["geometry"] = gdf.geometry.apply(
            lambda g: shapely_scale(g, xfact=sx, yfact=sy, origin=(0, 0))
        )

    # ---- Buffer the lines in pixel space -----------------------------
    # All coords are now in pixel space; drop any geographic CRS to
    # silence the "buffer in geographic CRS" warning from geopandas.
    gdf = gdf.set_crs(None, allow_override=True)
    gdf["geometry"] = gdf.geometry.buffer(buffer_px, cap_style=2)

    # ---- Rasterize ---------------------------------------------------
    # Build a pixel-space affine: identity, since everything is in pixel coords
    pixel_transform = rasterio.transform.from_bounds(
        0, 0, tile_size, tile_size, tile_size, tile_size
    )

    shapes = [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty]
    if not shapes:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)

    mask = rasterize(
        shapes,
        out_shape=(tile_size, tile_size),
        transform=pixel_transform,
        fill=0,
        dtype=np.uint8,
    )
    # Binary mask: 0 = background, 255 = road
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess(
    img_dir: Path,
    vec_dir: Path,
    out_dir: Path,
    tile_size: int = 512,
    buffer_px: int = 4,
) -> None:
    """
    End-to-end preprocessing: match images ↔ GeoJSONs, rasterize roads,
    and write 512×512 PNG pairs.

    Output structure
    ----------------
    out_dir/
        images/       <image_id>.png   – 3-channel RGB, uint8 [0, 255]
        masks/        <image_id>.png   – 1-channel binary, uint8 {0, 255}
    """
    out_images = out_dir / "images"
    out_masks = out_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # ---- Build lookup indices ----------------------------------------
    img_index = _build_index(img_dir, (".tif", ".tiff"))
    vec_index = _build_index(vec_dir, (".geojson", ".json"))

    logger.info(
        "Found %d image(s) and %d vector file(s)", len(img_index), len(vec_index)
    )

    all_ids = set(img_index.keys()) | set(vec_index.keys())
    matched = 0
    skipped = 0

    for image_id in sorted(all_ids):
        img_path = img_index.get(image_id)
        vec_path = vec_index.get(image_id)

        # ---- Handle missing pairs ------------------------------------
        if img_path is None:
            logger.warning(
                "⚠  No image found for vector '%s' (%s) – skipping",
                image_id,
                vec_path.name if vec_path else "?",
            )
            skipped += 1
            continue
        if vec_path is None:
            logger.warning(
                "⚠  No vector found for image '%s' (%s) – skipping",
                image_id,
                img_path.name if img_path else "?",
            )
            skipped += 1
            continue

        logger.info("Processing pair: %s  ↔  %s", img_path.name, vec_path.name)

        # ---- Read & resize image -------------------------------------
        img_arr = _read_image_as_array(img_path, tile_size)
        if img_arr is None:
            logger.error("✗  Could not read image %s – skipping", img_path.name)
            skipped += 1
            continue

        # ---- Rasterize mask ------------------------------------------
        mask_arr = _rasterize_roads(vec_path, img_path, tile_size, buffer_px)
        if mask_arr is None:
            logger.error("✗  Mask rasterization failed for %s – skipping", vec_path.name)
            skipped += 1
            continue

        # ---- Save as PNG pairs ---------------------------------------
        out_name = f"{image_id}.png"

        Image.fromarray(img_arr).save(out_images / out_name)
        Image.fromarray(mask_arr, mode="L").save(out_masks / out_name)

        matched += 1
        logger.info("  ✓  Saved %s  (%d×%d image, %d×%d mask)",
                     out_name, *img_arr.shape[:2], *mask_arr.shape)

    # ---- Summary -----------------------------------------------------
    logger.info("=" * 60)
    logger.info(
        "Done.  Matched: %d | Skipped: %d | Total IDs: %d",
        matched, skipped, len(all_ids),
    )
    logger.info("Output directory: %s", out_dir.resolve())


# ---------------------------------------------------------------------------
# PyTorch Dataset (drop-in ready)
# ---------------------------------------------------------------------------

class SpaceNetRoadDataset:
    """
    Minimal PyTorch-compatible Dataset for the processed image/mask pairs.

    Usage
    -----
    >>> from torch.utils.data import DataLoader
    >>> ds = SpaceNetRoadDataset("data/processed")
    >>> loader = DataLoader(ds, batch_size=8, shuffle=True)
    >>> for images, masks in loader:
    ...     # images: (B, 3, 512, 512) float32 [0, 1]
    ...     # masks:  (B, 1, 512, 512) float32 {0, 1}
    ...     pass
    """

    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"

        # Collect paired filenames
        img_names = {p.name for p in self.image_dir.glob("*.png")}
        msk_names = {p.name for p in self.mask_dir.glob("*.png")}
        self.filenames = sorted(img_names & msk_names)

        if not self.filenames:
            raise FileNotFoundError(
                f"No matching image/mask PNG pairs in {self.root}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]

        # Load image: (H, W, 3) uint8 → (3, H, W) float32 [0, 1]
        img = np.array(Image.open(self.image_dir / fname).convert("RGB"))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW

        # Load mask: (H, W) uint8 → (1, H, W) float32 {0, 1}
        msk = np.array(Image.open(self.mask_dir / fname).convert("L"))
        msk = (msk > 127).astype(np.float32)
        msk = msk[np.newaxis, ...]  # add channel dim

        if self.transform:
            img, msk = self.transform(img, msk)

        return img, msk


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SpaceNet 3 road-mask preprocessing pipeline",
    )
    p.add_argument(
        "--img_dir",
        type=Path,
        default=Path("data/sample_images"),
        help="Directory containing GeoTIFF satellite images",
    )
    p.add_argument(
        "--vec_dir",
        type=Path,
        default=Path("data/sample_geojson"),
        help="Directory containing GeoJSON road vector files",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for 512×512 PNG pairs",
    )
    p.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Output tile dimension in pixels (default: 512)",
    )
    p.add_argument(
        "--buffer_px",
        type=int,
        default=4,
        help="Road buffer half-width in pixels (default: 4)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess(
        img_dir=args.img_dir,
        vec_dir=args.vec_dir,
        out_dir=args.out_dir,
        tile_size=args.tile_size,
        buffer_px=args.buffer_px,
    )
