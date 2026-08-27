"""
model.py — Industry-Grade Road Segmentation Models
=====================================================
Provides production-quality segmentation architectures for the NetSight
pipeline, incorporating insights from OlmoEarth (Allen AI) research on
multi-scale feature fusion and masked-image-modeling pre-training.

Key Components
--------------
1.  RoadSegModel        — Primary model: UNet++ with EfficientNet-B4 encoder,
                          multi-class output (normal / critical / destroyed)
2.  TestTimeAugmentor   — TTA wrapper: flip-based ensemble for robust inference
3.  SlidingWindowInfer  — Arbitrary-size input via overlapping tile inference
4.  ResNetUNet          — Legacy wrapper (backward-compatible)
5.  OcclusionRobustModel— Legacy wrapper (backward-compatible)

Architecture rationale (OlmoEarth-inspired):
  OlmoEarth uses ViT encoder → decoder with multi-scale feature fusion.
  We mirror this with UNet++ (dense skip connections = multi-scale fusion)
  + MiT-B3 or EfficientNet-B4 encoder (strong pre-trained features).
  The UNet++ nested decoder is the closest practical equivalent to
  OlmoEarth's multi-resolution prediction heads.

Author : NetSight Project (ISRO NNRMS / Disaster Management Framework)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ══════════════════════════════════════════════════════════════════════════
# 1. PRIMARY MODEL — RoadSegModel
# ══════════════════════════════════════════════════════════════════════════

class RoadSegModel(nn.Module):
    """
    Industry-grade road segmentation model.

    Architecture: UNet++ with dense nested skip connections for multi-scale
    feature fusion (mirrors OlmoEarth's multi-resolution decoder approach).

    Encoder options:
      - 'efficientnet-b4' (default): 19M params, excellent accuracy/speed
      - 'mit_b3': MixVisionTransformer, 45M params, global context via SA
      - 'resnet50': 25M params, proven baseline

    Parameters
    ----------
    encoder_name : str
        Backbone encoder. See segmentation_models_pytorch docs.
    encoder_weights : str or None
        Pre-trained weights ('imagenet' or None).
    in_channels : int
        Number of input channels (3 for RGB).
    classes : int
        Number of output classes. Use 1 for binary (road/background),
        3 for multi-class (normal/critical/destroyed).
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()
        self.classes = classes

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # raw logits — we apply sigmoid/softmax outside
            decoder_attention_type="scse",  # Squeeze-and-Excitation for channel + spatial attention
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (B, C, H, W)."""
        return self.model(x)


# ══════════════════════════════════════════════════════════════════════════
# 2. TEST-TIME AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════

class TestTimeAugmentor:
    """
    Test-Time Augmentation wrapper for segmentation models.

    Runs inference on the original image plus geometric transformations
    (horizontal flip, vertical flip, both flips), then averages the
    predictions for a more robust output.

    Usage
    -----
    >>> model = RoadSegModel().eval()
    >>> tta = TestTimeAugmentor(model, device='cpu')
    >>> probs = tta.predict(image_tensor)  # (1, C, H, W) float [0,1]
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run TTA inference.

        Parameters
        ----------
        x : torch.Tensor (B, C, H, W)
            Normalized input tensor.

        Returns
        -------
        probs : torch.Tensor (B, Classes, H, W)
            Averaged probability map (sigmoid for binary, softmax for multi-class).
        """
        x = x.to(self.device)
        classes = self.model.classes if hasattr(self.model, 'classes') else 1

        transforms = [
            lambda t: t,                                        # original
            lambda t: torch.flip(t, dims=[-1]),                 # horizontal flip
            lambda t: torch.flip(t, dims=[-2]),                 # vertical flip
            lambda t: torch.flip(t, dims=[-2, -1]),             # both flips
        ]

        inverse_transforms = [
            lambda t: t,
            lambda t: torch.flip(t, dims=[-1]),
            lambda t: torch.flip(t, dims=[-2]),
            lambda t: torch.flip(t, dims=[-2, -1]),
        ]

        preds = []
        for fwd, inv in zip(transforms, inverse_transforms):
            augmented = fwd(x)
            logits = self.model(augmented)
            logits = inv(logits)

            if classes == 1:
                prob = torch.sigmoid(logits)
            else:
                prob = torch.softmax(logits, dim=1)
            preds.append(prob)

        return torch.stack(preds).mean(dim=0)


# ══════════════════════════════════════════════════════════════════════════
# 3. SLIDING-WINDOW INFERENCE (for arbitrary-size images)
# ══════════════════════════════════════════════════════════════════════════

class SlidingWindowInfer:
    """
    Sliding-window inference for large satellite images.

    Splits the input into overlapping tiles, runs inference on each,
    and stitches results using Gaussian-weighted blending to avoid
    seam artifacts at tile boundaries.

    Parameters
    ----------
    model : nn.Module
        Segmentation model.
    tile_size : int
        Size of each inference tile (default 512).
    overlap : int
        Overlap between adjacent tiles in pixels (default 128).
    device : str
        Compute device.
    use_tta : bool
        Whether to use TTA on each tile.
    """

    def __init__(
        self,
        model: nn.Module,
        tile_size: int = 512,
        overlap: int = 128,
        device: str = "cpu",
        use_tta: bool = False,
    ):
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = torch.device(device)
        self.use_tta = use_tta

        if use_tta:
            self.tta = TestTimeAugmentor(model, device)

        self.model.to(self.device)
        self.model.eval()

        # Pre-compute Gaussian weight kernel for blending
        self._weight = self._gaussian_weight(tile_size).to(self.device)

    @staticmethod
    def _gaussian_weight(size: int, sigma_scale: float = 0.25) -> torch.Tensor:
        """Create a 2D Gaussian weight map for tile blending."""
        sigma = size * sigma_scale
        center = size / 2.0
        y = torch.arange(size, dtype=torch.float32)
        x = torch.arange(size, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        gaussian = torch.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))
        return gaussian

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        normalize_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        """
        Run sliding-window inference on a potentially large image.

        Parameters
        ----------
        x : torch.Tensor (1, C, H, W)
            Input image tensor (pre-normalized or raw).
        normalize_fn : callable, optional
            Function to normalize each tile before inference.

        Returns
        -------
        output : torch.Tensor (1, Classes, H, W)
            Probability map for the full image.
        """
        _, C, H, W = x.shape
        ts = self.tile_size
        stride = ts - self.overlap

        # Pad image to fit tile grid
        pad_h = (math.ceil((H - ts) / stride) * stride + ts) - H if H > ts else ts - H
        pad_w = (math.ceil((W - ts) / stride) * stride + ts) - W if W > ts else ts - W
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x_padded.shape

        classes = self.model.classes if hasattr(self.model, 'classes') else 1
        output_sum = torch.zeros(1, classes, Hp, Wp, device=self.device)
        weight_sum = torch.zeros(1, 1, Hp, Wp, device=self.device)

        weight_tile = self._weight.unsqueeze(0).unsqueeze(0)  # (1, 1, ts, ts)

        for y in range(0, Hp - ts + 1, stride):
            for xp in range(0, Wp - ts + 1, stride):
                tile = x_padded[:, :, y:y + ts, xp:xp + ts].to(self.device)

                if normalize_fn is not None:
                    tile = normalize_fn(tile)

                if self.use_tta:
                    probs = self.tta.predict(tile)
                else:
                    logits = self.model(tile)
                    if classes == 1:
                        probs = torch.sigmoid(logits)
                    else:
                        probs = torch.softmax(logits, dim=1)

                output_sum[:, :, y:y + ts, xp:xp + ts] += probs * weight_tile
                weight_sum[:, :, y:y + ts, xp:xp + ts] += weight_tile

        # Avoid division by zero
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        output = output_sum / weight_sum

        # Crop back to original size
        return output[:, :, :H, :W]


# ══════════════════════════════════════════════════════════════════════════
# 4. LEGACY MODELS (Backward-compatible)
# ══════════════════════════════════════════════════════════════════════════

class ResNetUNet(nn.Module):
    """
    Industry-grade UNet with a ResNet-34 backbone pre-trained on ImageNet.
    Guarantees fast convergence and excellent feature extraction for roads.
    """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
        super().__init__()
        self.classes = classes
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class OcclusionRobustModel(nn.Module):
    """
    Advanced model architecture designed to handle occlusions (trees, shadows, clouds).
    Uses UnetPlusPlus with a MixVisionTransformer (mit_b3) encoder or MAnet.
    Provides multi-scale feature fusion and global context for long-range dependencies.
    """
    def __init__(self, architecture="unetplusplus", encoder_name="mit_b3", encoder_weights="imagenet", in_channels=3, classes=1):
        super().__init__()
        self.classes = classes
        if architecture.lower() == "unetplusplus":
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture.lower() == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        elif architecture.lower() == "manet":
            # MAnet uses Multi-scale Attention
            self.model = smp.MAnet(
                encoder_name="resnet50", # manet often uses strong ResNet
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ══════════════════════════════════════════════════════════════════════════
# 5. UTILITY — Generate Color-Coded Segmentation Mask
# ══════════════════════════════════════════════════════════════════════════

def generate_colored_mask(
    binary_mask: np.ndarray,
    criticality_map: Optional[np.ndarray] = None,
    destroyed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate a black-background color-coded road segmentation image.

    Color scheme:
      - Black (0, 0, 0)       → background
      - White (255, 255, 255)  → normal road
      - Yellow (255, 255, 0)   → critical road (high centrality)
      - Red (255, 0, 0)        → destroyed road

    Parameters
    ----------
    binary_mask : np.ndarray (H, W)
        Binary road mask. Non-zero = road.
    criticality_map : np.ndarray (H, W), optional
        Float array [0, 1] indicating road criticality per pixel.
        Pixels above 0.7 are marked yellow.
    destroyed_mask : np.ndarray (H, W), optional
        Binary mask indicating destroyed road pixels. Non-zero = destroyed.

    Returns
    -------
    colored : np.ndarray (H, W, 3) uint8
        RGB image with the color-coded road overlay on black background.
    """
    H, W = binary_mask.shape[:2]
    colored = np.zeros((H, W, 3), dtype=np.uint8)

    road_pixels = binary_mask > 0

    # Default: all roads are white
    colored[road_pixels] = [255, 255, 255]

    # Overlay critical roads in yellow
    if criticality_map is not None:
        critical_pixels = road_pixels & (criticality_map > 0.7)
        colored[critical_pixels] = [255, 255, 0]

    # Overlay destroyed roads in red (highest priority)
    if destroyed_mask is not None:
        destroyed_pixels = road_pixels & (destroyed_mask > 0)
        colored[destroyed_pixels] = [255, 0, 0]

    return colored


def mask_from_graph_criticality(
    binary_mask: np.ndarray,
    G,  # nx.Graph
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Create a criticality heatmap from graph edge betweenness centrality,
    projected back onto the mask pixel space.

    Parameters
    ----------
    binary_mask : np.ndarray (H, W)
        Binary road mask.
    G : networkx.Graph
        Road graph with 'path' edge attributes and node 'pos' attributes.
    shape : tuple (H, W)
        Output shape.

    Returns
    -------
    criticality_map : np.ndarray (H, W) float32 in [0, 1]
    """
    import networkx as nx

    H, W = shape
    crit_map = np.zeros((H, W), dtype=np.float32)

    if G.number_of_edges() == 0:
        return crit_map

    # Compute edge betweenness centrality
    edge_bc = nx.edge_betweenness_centrality(G, weight='w')

    # Normalize to [0, 1]
    max_bc = max(edge_bc.values()) if edge_bc else 1.0
    if max_bc < 1e-10:
        max_bc = 1.0

    for (u, v), bc in edge_bc.items():
        norm_bc = bc / max_bc
        path = G[u][v].get('path', [])
        for r, c in path:
            r, c = int(r), int(c)
            if 0 <= r < H and 0 <= c < W:
                crit_map[r, c] = max(crit_map[r, c], norm_bc)

    return crit_map