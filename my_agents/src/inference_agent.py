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

from src.model import ViTUNet
from src.mask_to_graph import mask_to_graph

# ── 2. BULLETPROOF MODEL INITIALIZATION ───────────────────────────────────
# We use two distinct variables here to prevent PyTorch from getting confused
device_str = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DEVICE = torch.device(device_str)

print(f"📦 Initializing ViT-UNet on {device_str.upper()}...", flush=True)

model = ViTUNet(base_ch=64, vit_depth=4, vit_heads=8)
weights_file = root_path / "weights" / "model_weights.pth"

# Load using the string, then move using the object. Flawless execution.
model.load_state_dict(torch.load(weights_file, map_location=device_str, weights_only=True))
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
        with rasterio.open(image_path) as src:
            # Safely grab up to 3 bands (prevents crashes on grayscale images)
            bands = min(src.count, 3)
            image = src.read(list(range(1, bands + 1)))
            
            # If the image is missing color channels, pad it for the neural net
            if bands == 1:
                image = np.repeat(image, 3, axis=0)
                
            image = np.transpose(image, (1, 2, 0)).astype(np.float32) / 255.0
            
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