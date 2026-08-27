"""
pseudo_labeler.py
=================
Generates pseudo-labels for Indian dataset screenshots by running
inference using the SpaceNet-pretrained model.

Extracts 512x512 patches, runs inference, applies a confidence threshold,
and saves the paired patches and masks for fine-tuning.
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_agents.src.model import OcclusionRobustModel

def main():
    # Directories
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "maps"))
    out_img_dir = os.path.join(raw_dir, "images")
    out_mask_dir = os.path.join(raw_dir, "masks")
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    # Check if raw images exist
    raw_images = glob.glob(os.path.join(raw_dir, "*.png"))
    
    if not raw_images:
        print(f"No PNG files found in {raw_dir}")
        return
        
    print(f"Found {len(raw_images)} raw screenshots.")
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "model_weights.pth"))
    if not os.path.exists(weights_path):
        print(f"ERROR: Teacher model weights not found at {weights_path}")
        return
        
    model = OcclusionRobustModel(
        architecture="unet", 
        encoder_name="resnet34", 
        encoder_weights=None, # we load our own
        in_channels=3, 
        classes=1
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Transform for inference (must match training normalization)
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    PATCH_SIZE = 512
    THRESHOLD = 0.6  # High threshold for pseudo-labels to ensure confidence
    
    total_patches = 0
    
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(raw_images, desc="Generating Pseudo-Labels")):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Failed to read {img_path}: {e}")
                continue
                
            w, h = img.size
            
            # Extract non-overlapping 512x512 patches
            for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
                for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                    patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                    patch_np = np.array(patch)
                    
                    # Skip mostly black/blank patches (e.g., edges of screenshots)
                    if patch_np.mean() < 10 or patch_np.mean() > 245:
                        continue
                        
                    # Preprocess for model
                    tensor = transform(image=patch_np)["image"].unsqueeze(0).to(device)
                    
                    # Inference
                    logits = model(tensor)
                    probs = torch.sigmoid(logits)
                    mask_np = probs.squeeze().cpu().numpy()
                    
                    # Apply confidence threshold
                    binary_mask = (mask_np > THRESHOLD).astype(np.uint8) * 255
                    
                    # If mask is completely empty, it might be background, but for fine-tuning 
                    # we want patches WITH roads to teach it Indian textures. 
                    # We'll keep patches with at least some road pixels.
                    if (binary_mask > 0).sum() < 500:
                        continue
                        
                    # Save paired patch and mask
                    patch_name = f"patch_{i:03d}_{x}_{y}"
                    patch.save(os.path.join(out_img_dir, f"{patch_name}.png"))
                    Image.fromarray(binary_mask).save(os.path.join(out_mask_dir, f"{patch_name}.png"))
                    
                    total_patches += 1
                    
                    # For speed during this phase, let's limit total patches generated
                    if total_patches >= 1000:
                        break
            if total_patches >= 1000:
                print("Generated enough patches for fine-tuning. Stopping early.")
                break

    print(f"\nDone! Generated {total_patches} high-quality (image, pseudo-mask) patch pairs for fine-tuning.")
    print(f"Saved to: {out_img_dir} and {out_mask_dir}")

if __name__ == "__main__":
    main()
