#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset, random_split

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
# Dataset Loader (Segment 2 Fixes Integrated)
# ---------------------------------------------------------------------------
class RoadSegDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[A.Compose] = None, pre_scanned_files: Optional[List[str]] = None):
        self.root = Path(root_dir)
        self.transform = transform
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "labels"

        if pre_scanned_files is not None:
            self.filenames = pre_scanned_files
        else:
            img_names = {Path(p).stem for p in self.image_dir.glob("*.tif")}
            msk_names = {Path(p).stem for p in self.mask_dir.glob("*.png")}
            self.filenames = sorted(img_names & msk_names)

        if not self.filenames:
            raise FileNotFoundError(f"No matching image/mask pairs found in {self.root}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]

        with rasterio.open(self.image_dir / f"{fname}.tif") as src:
            image = src.read([1, 2, 3])
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8)

        mask = np.array(Image.open(self.mask_dir / f"{fname}.png").convert("L"))
        mask = (mask > 127).astype(np.float32)

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
# Transforms & Augmentations (Segment 3 Fixes Integrated)
# ---------------------------------------------------------------------------
def get_train_transform(tile_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(tile_size, tile_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        # FIXED: Removed the invalid 'targets' keyword to comply with your runtime's Albumentations syntax
        A.CoarseDropout(num_holes_range=(4, 12), hole_height_range=(30, 80), hole_width_range=(30, 80), fill=0, p=0.7),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

def get_val_transform(tile_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(tile_size, tile_size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

# ---------------------------------------------------------------------------
# Neural Network Architecture (Segment 4)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), H, W

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTBottleneck(nn.Module):
    def __init__(self, embed_dim: int = 512, depth: int = 4, num_heads: int = 8, max_spatial: int = 32, drop: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_spatial * max_spatial, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens, H, W = self.patch_embed(x)
        N = H * W
        pe = self.pos_embed if self.pos_embed.shape[1] == N else F.interpolate(
            self.pos_embed.reshape(1, 32, 32, C).permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1).reshape(1, N, C)
        
        tokens = tokens + pe
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens).transpose(1, 2).reshape(B, C, H, W)

class ViTUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_ch: int = 64, vit_depth: int = 4, vit_heads: int = 8, drop: float = 0.1):
        super().__init__()
        ch = base_ch
        self.enc1 = ConvBlock(in_channels, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.enc4 = ConvBlock(ch * 4, ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ViTBottleneck(embed_dim=ch * 8, depth=vit_depth, num_heads=vit_heads, max_spatial=32, drop=drop)
        self.up4 = nn.ConvTranspose2d(ch * 8, ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(ch * 16, ch * 8)
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(ch * 8, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)
        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([e4, self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([e3, self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))
        return self.head(d1)

# ---------------------------------------------------------------------------
# Soft Skeletonisation & Loss Engine (Segment 5 Fixes Integrated)
# ---------------------------------------------------------------------------
def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    if img.shape[-1] <= 1 or img.shape[-2] <= 1: return img
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    if img.shape[-1] <= 1 or img.shape[-2] <= 1: return img
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))

def _soft_skel(img: torch.Tensor, iters: int = 15) -> torch.Tensor:
    img_open = _soft_open(img)
    skel = F.relu(img - img_open)
    remaining = _soft_erode(img)
    for _ in range(iters - 1):
        remaining_open = _soft_open(remaining)
        delta = F.relu(remaining - remaining_open)
        skel = skel + delta
        remaining = _soft_erode(remaining)
    return torch.clamp(skel, 0.0, 1.0)

class CLDiceLoss(nn.Module):
    def __init__(self, skel_iters: int = 15, smooth: float = 1.0):
        super().__init__()
        self.iters = skel_iters
        self.smooth = smooth
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        skel_pred = _soft_skel(probs, self.iters)
        skel_gt = _soft_skel(targets, self.iters)
        tprec_num = (skel_pred * targets).sum(dim=(2, 3)) + self.smooth
        tprec_den = skel_pred.sum(dim=(2, 3)) + self.smooth
        tsens_num = (skel_gt * probs).sum(dim=(2, 3)) + self.smooth
        tsens_den = skel_gt.sum(dim=(2, 3)) + self.smooth
        return (1.0 - (2.0 * (tprec_num / tprec_den) * (tsens_num / tsens_den)) / ((tprec_num / tprec_den) + (tsens_num / tsens_den) + 1e-8)).mean()

class HybridLoss(nn.Module):
    def __init__(self, skel_iters: int = 15):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.cldice = CLDiceLoss(skel_iters=skel_iters)
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce_loss = self.bce(logits, targets)
        cld_loss = self.cldice(logits, targets)
        return 0.5 * bce_loss + 0.5 * cld_loss, bce_loss, cld_loss

# ---------------------------------------------------------------------------
# Metrics Tracker (Segment 6 Fixes Integrated)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    return ((intersection + 1e-7) / (union + 1e-7)).mean().item()

# ---------------------------------------------------------------------------
# Training Infrastructure (Segment 7 Validation Tweaks Integrated)
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: HybridLoss, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device) -> Tuple[float, float, float, float]:
    model.train()
    running_loss, running_bce, running_cld, running_iou, n_batches = 0.0, 0.0, 0.0, 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            total_loss, bce_loss, cld_loss = criterion(logits, masks)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += total_loss.item()
        running_bce += bce_loss.item()
        running_cld += cld_loss.item()
        running_iou += compute_iou(logits, masks)
        n_batches += 1
    n = max(n_batches, 1)
    return running_loss / n, running_bce / n, running_cld / n, running_iou / n

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: HybridLoss, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    running_loss, running_bce, running_cld, running_iou, n_batches = 0.0, 0.0, 0.0, 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            total_loss, bce_loss, cld_loss = criterion(logits, masks)
        running_loss += total_loss.item()
        running_bce += bce_loss.item()
        running_cld += cld_loss.item()
        running_iou += compute_iou(logits, masks)
        n_batches += 1
    n = max(n_batches, 1)
    return running_loss / n, running_bce / n, running_cld / n, running_iou / n

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    
    # Load Master Data Layout Safely
    full_dataset = RoadSegDataset(args.data_dir, transform=get_train_transform(args.tile_size))
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    train_ds, val_ds = random_split(full_dataset, [n_total - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))
    
    # FIXED: Re-initialize validation file lists cleanly to bypass directory glob warnings
    val_files = [full_dataset.filenames[i] for i in val_ds.indices]
    val_ds_wrapper = RoadSegDataset(args.data_dir, transform=get_val_transform(args.tile_size), pre_scanned_files=val_files)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds_wrapper, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = ViTUNet(in_channels=3, out_channels=1, base_ch=args.base_ch, vit_depth=args.vit_depth, vit_heads=args.vit_heads, drop=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = HybridLoss(skel_iters=args.skel_iters).to(device)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_bce, tr_cld, tr_iou = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl_loss, vl_bce, vl_cld, vl_iou = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        log.info(
            f"Epoch {epoch:3d}/{args.epochs} [{time.time()-t0:.1f}s] lr={optimizer.param_groups[0]['lr']:.2e} "
            f"Train loss={tr_loss:.4f} (bce={tr_bce:.4f} cld={tr_cld:.4f}) IoU={tr_iou:.4f} | "
            f"Val loss={vl_loss:.4f} (bce={vl_bce:.4f} cld={vl_cld:.4f}) IoU={vl_iou:.4f}"
            f"{' ★' if vl_iou > best_iou else ''}"
        )
        if vl_iou > best_iou:
            best_iou = vl_iou
            torch.save(model.state_dict(), args.save_path)

    log.info(f"Training Complete! Best Val IoU: {best_iou:.4f} -> Weights saved to {args.save_path}")

# ---------------------------------------------------------------------------
# Colab Jupyter Launcher Pipeline Configuration (Argument Parser Safety Patch)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="spacenet_data")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--vit_depth", type=int, default=4)
    parser.add_argument("--vit_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--skel_iters", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="model_weights.pth")
    
    # FIXED: Passing [] bypasses the hidden -f kernel flag injected by Google Colab's backend core
    args = parser.parse_args([])
    main(args)