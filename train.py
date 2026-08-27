#!/usr/bin/env python3
"""
train.py — Industry-Grade Road Segmentation Training Pipeline
==============================================================
Production training script for NetSight road segmentation models.

Incorporates lessons from OlmoEarth (Allen AI) research:
  - Masked-patch-style augmentation for spatial context learning
  - Multi-scale feature fusion via UNet++ decoder
  - Proper loss balancing for thin-structure (road) segmentation

Key Features
------------
- Hybrid Loss: BCE + Dice + Soft-clDice (centerline-aware)
- EMA (Exponential Moving Average) model for stable predictions
- Gradient accumulation for effective larger batch sizes
- Cosine annealing with linear warm-up
- Proper augmentation pipeline (moderate, not destructive)
- Mixed-precision training (AMP) for GPU efficiency

Author: NetSight Project (ISRO NNRMS / Disaster Management Framework)
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("netsight.train")

# ---------------------------------------------------------------------------
# Dataset Loader (PNG + TIFF support)
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")

class RoadSegDataset(Dataset):
    """
    Road segmentation dataset loader supporting GeoTIFF and PNG formats.

    Expects directory structure:
        root_dir/
            images/   <stem>.<ext>
            masks/    <stem>.png

    Masks are loaded as grayscale and binarized at threshold 127.
    """
    def __init__(self, root_dir: str, transform: Optional[A.Compose] = None,
                 pre_scanned_files: Optional[List[str]] = None):
        self.root = Path(root_dir)
        self.transform = transform
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir.resolve()} (Ensure dataset is extracted and --data_dir is correct)")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir.resolve()} (Ensure dataset is extracted and --data_dir is correct)")

        if pre_scanned_files is not None:
            self.filenames = pre_scanned_files
        else:
            # Scan for all supported image formats
            img_names = set()
            for ext in _IMAGE_EXTENSIONS:
                img_names |= {Path(p).stem for p in self.image_dir.glob(ext)}
            # Masks can be PNG or any supported format
            msk_names = set()
            for ext in ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"):
                msk_names |= {Path(p).stem for p in self.mask_dir.glob(ext)}
            self.filenames = sorted(img_names & msk_names)

        # Build a stem→extension lookup for fast loading
        self._img_ext_map: dict[str, str] = {}
        for fname in self.filenames:
            for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                if (self.image_dir / f"{fname}{ext}").exists():
                    self._img_ext_map[fname] = ext
                    break

        self._msk_ext_map: dict[str, str] = {}
        for fname in self.filenames:
            for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
                if (self.mask_dir / f"{fname}{ext}").exists():
                    self._msk_ext_map[fname] = ext
                    break

        if not self.filenames:
            raise FileNotFoundError(f"No matching image/mask pairs found in {self.root}")

    def __len__(self) -> int:
        return len(self.filenames)

    def _load_image(self, fname: str) -> np.ndarray:
        """Load an image as (H, W, 3) uint8 array, supporting both GeoTIFF and PNG."""
        ext = self._img_ext_map.get(fname, ".tif")
        img_path = self.image_dir / f"{fname}{ext}"

        if ext in (".tif", ".tiff"):
            # GeoTIFF: use rasterio for proper band handling
            with rasterio.open(img_path) as src:
                bands = min(src.count, 3)
                image = src.read(list(range(1, bands + 1)))
                if bands == 1:
                    image = np.repeat(image, 3, axis=0)

                # Handle 16-bit data properly
                if image.dtype == np.uint16 or image.max() > 255:
                    p2, p98 = np.percentile(image, (2, 98))
                    image = np.clip(image, p2, p98)
                    image = ((image - p2) / (p98 - p2 + 1e-8) * 255.0).astype(np.uint8)

                image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        else:
            # PNG/JPG (and other PIL-readable formats)
            image = np.array(Image.open(img_path).convert("RGB"))

        return image

    def _load_mask(self, fname: str) -> np.ndarray:
        """Load a mask as (H, W) float32 array, binarized."""
        ext = self._msk_ext_map.get(fname, ".png")
        mask_path = self.mask_dir / f"{fname}{ext}"
        mask = np.array(Image.open(mask_path).convert("L"))
        # Robust binarization: handle both 0/255 and 0/1 masks
        mask = (mask > 127).astype(np.float32)
        return mask

    def get_road_density(self, idx: int) -> float:
        """Return the fraction of road pixels in the mask (for weighted sampling)."""
        fname = self.filenames[idx]
        mask = self._load_mask(fname)
        return float(mask.sum()) / (mask.shape[0] * mask.shape[1] + 1e-8)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]

        image = self._load_image(fname)
        mask = self._load_mask(fname)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

# ---------------------------------------------------------------------------
# Transforms & Augmentations
# ---------------------------------------------------------------------------
# ImageNet statistics for proper input normalization
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transform(tile_size: int = 512) -> A.Compose:
    """
    Training augmentation pipeline.

    Design rationale (OlmoEarth-inspired):
    - Geometric transforms: flips + rotation for orientation invariance
    - Color transforms: brightness/contrast/hue for lighting invariance
    - CLAHE: enhances road visibility in shadowed regions
    - GridDropout: OlmoEarth-style patch masking for spatial context learning
    - CoarseDropout: simulates tree canopy/cloud occlusion (MODERATE, not destructive)
    """
    return A.Compose([
        A.Resize(tile_size, tile_size),
        # Geometric augmentation
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15),
            border_mode=0, p=0.4,
        ),
        # Color augmentation
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        # Weather/condition simulation
        A.OneOf([
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=1.0),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=1.0),
        ], p=0.15),
        # OlmoEarth-inspired: patch masking for spatial context learning
        A.GridDropout(ratio=0.2, random_offset=True, holes_number_xy=(4, 4), p=0.2),
        # Moderate occlusion simulation (tree canopy / small clouds)
        A.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(20, 50),
            hole_width_range=(20, 50),
            fill=0, p=0.3,
        ),
        # Normalize and convert
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transform(tile_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(tile_size, tile_size),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
import segmentation_models_pytorch as smp

def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, (3,3), (1,1), (1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_=3):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

class SoftCLDiceLoss(nn.Module):
    """Soft Centerline Dice Loss for topology-preserving road segmentation."""
    def __init__(self, iter_=3, smooth=1.):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        y_pred = torch.sigmoid(logits)
        y_true = targets
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens + 1e-7)
        return cl_dice

class HybridLoss(nn.Module):
    """
    Hybrid loss combining BCE, Dice, and Centerline-Dice for road segmentation.

    The balanced weights ensure:
    - BCE: pixel-level accuracy (handles class imbalance with pos_weight)
    - Dice: region overlap optimization (critical for road area matching)
    - CLDice: topology preservation (ensures road continuity/connectivity)
    """
    def __init__(self, bce_weight: float = 0.3, dice_weight: float = 0.4, cldice_weight: float = 0.3):
        super().__init__()
        self.bce = smp.losses.SoftBCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.cldice = SoftCLDiceLoss(iter_=3)
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.cldice_w = cldice_weight

        # Validate weights sum to ~1.0
        total = bce_weight + dice_weight + cldice_weight
        if abs(total - 1.0) > 0.01:
            log.warning(f"Loss weights sum to {total:.3f}, not 1.0. Consider adjusting.")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        cldice_loss = self.cldice(logits, targets)
        total_loss = self.bce_w * bce_loss + self.dice_w * dice_loss + self.cldice_w * cldice_loss
        return total_loss, bce_loss, dice_loss, cldice_loss

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    return ((intersection + 1e-7) / (union + 1e-7)).mean().item()

@torch.no_grad()
def compute_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1.mean().item()

# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) Model
# ---------------------------------------------------------------------------
class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that is updated with
    a moving average at each step. The EMA model typically produces
    smoother, more generalizable predictions than the raw trained model.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)

# ---------------------------------------------------------------------------
# Learning Rate Scheduler with Warm-up
# ---------------------------------------------------------------------------
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warm-up."""
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cos_factor for base_lr in self.base_lrs]

# ---------------------------------------------------------------------------
# Training Infrastructure
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: HybridLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
    ema: Optional[EMAModel] = None,
) -> Tuple[float, float, float, float, float, float]:
    """Train for one epoch with gradient accumulation and EMA updates."""
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_cld = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            total_loss, bce_loss, dice_loss, cldice_loss = criterion(logits, masks)
            # Scale loss for gradient accumulation
            scaled_loss = total_loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # EMA update
            if ema is not None:
                ema.update(model)

        running_loss += total_loss.item()
        running_bce += bce_loss.item()
        running_dice += dice_loss.item()
        running_cld += cldice_loss.item()
        running_iou += compute_iou(logits, masks)
        running_f1 += compute_f1(logits, masks)
        n_batches += 1

    n = max(n_batches, 1)
    return (running_loss / n, running_bce / n, running_dice / n,
            running_cld / n, running_iou / n, running_f1 / n)

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: HybridLoss,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float]:
    """Validate the model on the validation set."""
    model.eval()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_cld = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            total_loss, bce_loss, dice_loss, cldice_loss = criterion(logits, masks)

        running_loss += total_loss.item()
        running_bce += bce_loss.item()
        running_dice += dice_loss.item()
        running_cld += cldice_loss.item()
        running_iou += compute_iou(logits, masks)
        running_f1 += compute_f1(logits, masks)
        n_batches += 1

    n = max(n_batches, 1)
    return (running_loss / n, running_bce / n, running_dice / n,
            running_cld / n, running_iou / n, running_f1 / n)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    full_dataset = RoadSegDataset(args.data_dir, transform=get_train_transform(args.tile_size))
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Proper validation dataset with val-only transforms (no augmentation)
    val_files = [full_dataset.filenames[i] for i in val_ds.indices]
    val_ds_clean = RoadSegDataset(
        args.data_dir, transform=get_val_transform(args.tile_size),
        pre_scanned_files=val_files,
    )

    log.info(f"Dataset: {n_total} total → {n_train} train + {n_val} val")

    # Optional: weighted sampling to oversample road-dense tiles
    if args.weighted_sampling:
        log.info("Computing road density weights for balanced sampling...")
        densities = []
        for i in train_ds.indices:
            densities.append(full_dataset.get_road_density(i))
        # Weight = density + small offset (so empty tiles still appear occasionally)
        weights = [d + 0.1 for d in densities]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds_clean, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Model selection
    from my_agents.src.model import RoadSegModel, ResNetUNet, OcclusionRobustModel

    if args.model_type == "industry":
        model = RoadSegModel(
            encoder_name=args.encoder_name,
            encoder_weights="imagenet",
            in_channels=3, classes=1,
        ).to(device)
    elif args.model_type == "resnet":
        model = ResNetUNet(
            encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=3, classes=1,
        ).to(device)
    else:
        model = OcclusionRobustModel(
            architecture=args.model_type, encoder_name="mit_b3",
            encoder_weights="imagenet", in_channels=3, classes=1,
        ).to(device)

    # Optional fine-tuning from existing weights
    if getattr(args, "fine_tune", False):
        log.info(f"Loading pre-trained weights from {args.pretrained_weights} for fine-tuning...")
        state = torch.load(args.pretrained_weights, map_location=device)
        model.load_state_dict(state, strict=False)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {args.model_type} | Params: {n_params:,} ({n_trainable:,} trainable)")

    # EMA
    ema = EMAModel(model, decay=args.ema_decay) if args.use_ema else None
    if ema:
        log.info(f"EMA enabled with decay={args.ema_decay}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs, eta_min=1e-6,
    )

    # Loss function with PROPER balanced weights
    criterion = HybridLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        cldice_weight=args.cldice_weight,
    ).to(device)
    log.info(f"Loss weights: BCE={args.bce_weight}, Dice={args.dice_weight}, CLDice={args.cldice_weight}")

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    best_iou = 0.0
    best_loss = float("inf")
    patience_counter = 0

    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_bce, tr_dice, tr_cld, tr_iou, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            max_grad_norm=args.max_grad_norm,
            accumulation_steps=args.accumulation_steps,
            ema=ema,
        )

        # Validate with EMA model if available, otherwise with training model
        eval_model = ema.shadow if ema else model
        vl_loss, vl_bce, vl_dice, vl_cld, vl_iou, vl_f1 = validate(
            eval_model, val_loader, criterion, device,
        )

        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        # Log with all metrics
        improved = ""
        if vl_iou > best_iou:
            improved = " ★ IoU"
        if vl_loss < best_loss:
            improved += " ★ Loss"

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} [{elapsed:.1f}s] lr={lr_now:.2e} "
            f"Train loss={tr_loss:.4f} (bce={tr_bce:.4f} dice={tr_dice:.4f} cld={tr_cld:.4f}) "
            f"IoU={tr_iou:.4f} F1={tr_f1:.4f} | "
            f"Val loss={vl_loss:.4f} (bce={vl_bce:.4f} dice={vl_dice:.4f} cld={vl_cld:.4f}) "
            f"IoU={vl_iou:.4f} F1={vl_f1:.4f}{improved}"
        )

        # Save best model by IoU
        if vl_iou > best_iou:
            best_iou = vl_iou
            patience_counter = 0
            save_state = ema.state_dict() if ema else model.state_dict()
            torch.save(save_state, args.save_path)
            log.info(f"  → Saved best-IoU model ({best_iou:.4f}) to {args.save_path}")

        # Save best model by loss (separate checkpoint)
        if vl_loss < best_loss:
            best_loss = vl_loss
            patience_counter = 0
            save_state = ema.state_dict() if ema else model.state_dict()
            loss_path = str(args.save_path).replace(".pth", "_best_loss.pth")
            torch.save(save_state, loss_path)
        else:
            patience_counter += 1

        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            log.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break

    log.info(f"Training Complete! Best Val IoU: {best_iou:.4f} → Weights saved to {args.save_path}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetSight Road Segmentation Training")

    # Data
    parser.add_argument("--data_dir", type=str, default="spacenet_data")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.2)

    # Model
    parser.add_argument("--model_type", type=str, default="industry",
                        choices=["industry", "resnet", "unetplusplus", "manet"])
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4",
                        help="Encoder backbone for 'industry' model type")

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size × accumulation_steps)")
    parser.add_argument("--early_stopping", type=int, default=15,
                        help="Stop after N epochs without improvement (0 = disabled)")

    # Loss weights (must sum to 1.0)
    parser.add_argument("--bce_weight", type=float, default=0.3)
    parser.add_argument("--dice_weight", type=float, default=0.4)
    parser.add_argument("--cldice_weight", type=float, default=0.3)

    # EMA
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Sampling
    parser.add_argument("--weighted_sampling", action="store_true", default=False,
                        help="Oversample tiles with high road density")

    # Misc
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="my_agents/weights/model_weights.pth")
    parser.add_argument("--fine_tune", action="store_true", default=False,
                        help="Load pre-trained weights for fine-tuning")
    parser.add_argument("--pretrained_weights", type=str, default="my_agents/weights/model_weights.pth",
                        help="Path to pre-trained weights for fine-tuning")

    # Colab compatibility
    import sys as _sys
    _in_colab = "google.colab" in _sys.modules
    _has_explicit_args = len(_sys.argv) > 1 and not _sys.argv[1].startswith("-f")
    args = parser.parse_args(None if _has_explicit_args or not _in_colab else [])
    main(args)