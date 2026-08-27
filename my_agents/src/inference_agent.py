import asyncio
import sys
import os
from pathlib import Path
import numpy as np
import torch
import rasterio
from google.antigravity import Agent, LocalAgentConfig

# ── 1. DYNAMIC PATH INJECTION ─────────────────────────────────────────────
root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.model import ResNetUNet
from src.mask_to_graph import mask_to_graph

# ── 2. MODEL INITIALIZATION ──────────────────────────────────────────────
device_str = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DEVICE = torch.device(device_str)

print(f"📦 Initializing ResNet-UNet on {device_str.upper()}...", flush=True)

model = ResNetUNet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
weights_file = root_path / "weights" / "model_weights.pth"

if not weights_file.exists():
    # Try legacy weights
    weights_file = root_path / "weights" / "model_weights_india_osm.pth"

if weights_file.exists():
    model.load_state_dict(torch.load(weights_file, map_location=device_str, weights_only=True), strict=False)
    print(f"✅ Loaded weights from {weights_file.name}", flush=True)
else:
    print(f"⚠️ No weights found. Model will use random initialization.", flush=True)

model.to(COMPUTE_DEVICE)
model.eval()

# ── 3. THE TOOL ───────────────────────────────────────────────────────────
def parse_satellite_geometry(image_path: str) -> str:
    """Parses a geospatial satellite tile through the ViT-UNet model."""
    
    # ⚠️ FUTURE-PROOFING: Checks if the image actually exists on your local PC
    if not os.path.exists(image_path):
        return (
            f"[ANALYSIS ERROR] File not found at '{image_path}'. "
            f"If you just moved from Colab, you need to download a sample .tif image "
            f"and place it in that folder!"
        )
        
    try:
        # Format-aware loading: try rasterio (GeoTIFF), fall back to PIL (PNG, JPG, etc.)
        image = None
        try:
            with rasterio.open(image_path) as src:
                # Safely grab up to 3 bands (prevents crashes on grayscale images)
                bands = min(src.count, 3)
                img_data = src.read(list(range(1, bands + 1)))
                
                # If the image is missing color channels, pad it for the neural net
                if bands == 1:
                    img_data = np.repeat(img_data, 3, axis=0)
                    
                image = np.transpose(img_data, (1, 2, 0)).astype(np.float32) / 255.0
        except Exception:
            # Fallback for PNG and other standard image formats
            from PIL import Image as PILImage
            pil_img = PILImage.open(image_path).convert("RGB")
            image = np.array(pil_img, dtype=np.float32) / 255.0
            
        input_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(COMPUTE_DEVICE)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().squeeze().numpy()
            
        binary_road_mask = (probs > 0.01).astype(np.uint8)
        continuity_score = float(probs.mean())
        detected_pixels = int(binary_road_mask.sum())
        
        return (
            f"[ANALYSIS SUCCESS] Tile processed: '{image_path}'\n"
            f"├── Global Continuity Score: {continuity_score:.4f}\n"
            f"└── Extracted Road Pixels: {detected_pixels:,} px"
        )
    except Exception as err:
        return f"[ANALYSIS ERROR] Failed to parse geometry: {str(err)}"

# ── 4. THE AGENT ORCHESTRATOR ─────────────────────────────────────────────
async def main():
    print("🔥 Registering optimized tools over the Antigravity Framework...", flush=True)
    
    api_key = os.environ.get("ANTIGRAVITY_API_KEY", "")
    if not api_key:
        print("❌ Error: ANTIGRAVITY_API_KEY environment variable is not set.", flush=True)
        sys.exit(1)

    config = LocalAgentConfig(
        api_key=api_key,
        model="gemini-3.1-flash-lite",      
        system_instructions=(
            "You are the central NetSight Routing Agent. Your primary objective "
            "is to verify road connectivity over geospatial regions. "
            "You have access to a 'parse_satellite_geometry' tool that analyzes satellite tiles "
            "using custom ViT-UNet weights."
        ),
        tools=[parse_satellite_geometry]
    )

    async with Agent(config) as agent:
        print("🚀 NetSight Perception Agent is active and listening.\n", flush=True)
        
        # We tell the agent to test a sample file. If it doesn't exist, our tool handles it!
        response = await agent.chat(
            "Verify your toolset. Run the geometry parser tool over a simulation "
            "sample file located at 'spacenet_data/images/sample_0.tif'."
        )
        print("🤖 Agent Response Trace:")
        print(await response.text())

if __name__ == "__main__":
    asyncio.run(main())