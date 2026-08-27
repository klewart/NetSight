"""
data_generator.py -- Industry-Grade Data Generation Pipeline
=============================================================
Downloads paired high-resolution satellite imagery (ESRI World Imagery)
and OpenStreetMap (OSM) vector data, rasterising the OSM roads into
properly-width binary ground-truth masks with road-type classification.

Incorporates lessons from OlmoEarth research:
  - Proper road width calibration per highway type
  - CLAHE enhancement for improved road visibility
  - Diverse geographic sampling across India

Designed for ISRO NetSight challenge (India-specific dataset).
"""

import io
import os
import time
import requests
import numpy as np
import mercantile
import osmnx as ox
import networkx as nx
from PIL import Image, ImageDraw, ImageFilter

# Configure OSMnx
ox.settings.log_console = False
ox.settings.use_cache = True

# Constants
ZOOM_LEVEL = 16  # Good balance of resolution and area (approx 2.4 meters/pixel at equator)
TILE_SIZE = 256
OUTPUT_SIZE = 512  # We stitch 2x2 tiles to get 512x512 images

# Directories
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "india_dataset"))
IMG_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# ── Road Width Calibration ────────────────────────────────────────────────
# At zoom level 16, ~2.4 m/px at equator. Indian roads typically:
#   - National Highway / Motorway: 12-20m wide → 5-8 px
#   - State Highway / Trunk: 8-14m wide → 3-6 px
#   - Primary Road: 6-10m wide → 3-4 px
#   - Secondary/Tertiary: 4-8m wide → 2-3 px
#   - Residential/Service: 3-5m wide → 1-2 px
# We use slightly wider than real to ensure the model can learn thin features.

ROAD_WIDTH_BY_TYPE = {
    "motorway": 10,
    "motorway_link": 7,
    "trunk": 9,
    "trunk_link": 6,
    "primary": 8,
    "primary_link": 6,
    "secondary": 7,
    "secondary_link": 5,
    "tertiary": 6,
    "tertiary_link": 4,
    "residential": 5,
    "living_street": 4,
    "service": 4,
    "unclassified": 5,
}
DEFAULT_ROAD_WIDTH = 5  # Fallback for unknown types

# ── Expanded Indian Cities (20+ cities for geographic diversity) ──────────
CITIES = {
    # Tier 1 Metros
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    # Tier 2 Cities
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Chandigarh": (30.7333, 76.7794),
    "Kochi": (9.9312, 76.2673),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Nagpur": (21.1458, 79.0882),
    "Coimbatore": (11.0168, 76.9558),
    # Challenging terrain
    "Shimla": (31.1048, 77.1734),       # Mountain roads
    "Guwahati": (26.1445, 91.7362),     # Northeast India
    "Visakhapatnam": (17.6868, 83.2185), # Coastal
    "Varanasi": (25.3176, 82.9739),     # Dense historic streets
}


def get_esri_tile(x: int, y: int, z: int) -> Image.Image:
    """Download a single 256x256 satellite tile from ESRI."""
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    headers = {"User-Agent": "NetSight-ISRO-Challenge"}

    for _ in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                if img.size == (TILE_SIZE, TILE_SIZE):
                    return img
        except Exception:
            pass
        time.sleep(1)

    # Return black tile if failed
    return Image.new("RGB", (TILE_SIZE, TILE_SIZE), color=(0, 0, 0))


def apply_clahe_enhancement(img: Image.Image) -> Image.Image:
    """
    Apply CLAHE-like contrast enhancement to improve road visibility.

    Uses a combination of PIL operations to approximate CLAHE without
    requiring OpenCV, ensuring compatibility across environments.
    """
    import numpy as np

    img_arr = np.array(img, dtype=np.float32)

    # Per-channel adaptive histogram stretching (2nd-98th percentile)
    for c in range(3):
        channel = img_arr[:, :, c]
        p2, p98 = np.percentile(channel, (2, 98))
        if p98 - p2 > 10:  # Only enhance if there's meaningful range
            channel = np.clip((channel - p2) / (p98 - p2) * 255.0, 0, 255)
            img_arr[:, :, c] = channel

    # Light unsharp mask for edge enhancement (makes roads more visible)
    enhanced = Image.fromarray(img_arr.astype(np.uint8))
    blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=2))
    blurred_arr = np.array(blurred, dtype=np.float32)
    enhanced_arr = img_arr + 0.3 * (img_arr - blurred_arr)
    enhanced_arr = np.clip(enhanced_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(enhanced_arr)


def latlon_to_pixel(lat: float, lon: float, bounds: mercantile.LngLatBbox, img_size: int) -> tuple[int, int]:
    """Convert lat/lon to pixel coordinates within the image bounds."""
    x_pct = (lon - bounds.west) / (bounds.east - bounds.west)
    # Latitude is inverted (north is top, so smaller Y)
    y_pct = (bounds.north - lat) / (bounds.north - bounds.south)

    px = int(x_pct * img_size)
    py = int(y_pct * img_size)
    return px, py


def get_road_width(highway_type: str) -> int:
    """Get the pixel width for a given OSM highway type."""
    if isinstance(highway_type, list):
        # OSM sometimes returns multiple types; use the most important
        for ht in highway_type:
            if ht in ROAD_WIDTH_BY_TYPE:
                return ROAD_WIDTH_BY_TYPE[ht]
        return DEFAULT_ROAD_WIDTH
    return ROAD_WIDTH_BY_TYPE.get(str(highway_type), DEFAULT_ROAD_WIDTH)


def generate_tile_pair(base_x: int, base_y: int, z: int, prefix: str, enhance: bool = True):
    """Generates a 512x512 image and mask by stitching 2x2 tiles."""
    img_name = os.path.join(IMG_DIR, f"{prefix}_z{z}_x{base_x}_y{base_y}.png")
    mask_name = os.path.join(MASK_DIR, f"{prefix}_z{z}_x{base_x}_y{base_y}.png")

    if os.path.exists(img_name) and os.path.exists(mask_name):
        return

    # 1. Fetch 2x2 Satellite Tiles
    stitched_img = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE))
    for dx in [0, 1]:
        for dy in [0, 1]:
            tile_img = get_esri_tile(base_x + dx, base_y + dy, z)
            stitched_img.paste(tile_img, (dx * TILE_SIZE, dy * TILE_SIZE))

    # Apply CLAHE-like enhancement for better road visibility
    if enhance:
        stitched_img = apply_clahe_enhancement(stitched_img)

    # 2. Calculate geographic bounds for the 2x2 area
    nw = mercantile.ul(base_x, base_y, z)
    se = mercantile.ul(base_x + 2, base_y + 2, z)

    # mercantile Bbox: left, bottom, right, top (west, south, east, north)
    bounds = mercantile.LngLatBbox(nw.lng, se.lat, se.lng, nw.lat)

    # 3. Fetch OSM Data for the bounds
    buffer = 0.005
    try:
        G = ox.graph_from_bbox(
            bbox=(bounds.west - buffer, bounds.south - buffer, bounds.east + buffer, bounds.north + buffer),
            network_type="drive",
            simplify=True
        )
    except Exception as e:
        print(f"  [!] Failed to fetch OSM data for {prefix}_{base_x}_{base_y}: {e}")
        # Save empty mask if no roads found
        mask_img = Image.new("L", (OUTPUT_SIZE, OUTPUT_SIZE), color=0)
        stitched_img.save(img_name)
        mask_img.save(mask_name)
        return

    # 4. Rasterize OSM graph into a mask with road-type-aware widths
    mask_img = Image.new("L", (OUTPUT_SIZE, OUTPUT_SIZE), color=0)
    draw = ImageDraw.Draw(mask_img)

    for u, v, data in G.edges(data=True):
        # Determine road width based on highway type
        highway_type = data.get("highway", "unclassified")
        road_width = get_road_width(highway_type)

        if "geometry" in data:
            # LineString with multiple points
            coords = list(data["geometry"].coords)
            points = []
            for lon, lat in coords:
                px, py = latlon_to_pixel(lat, lon, bounds, OUTPUT_SIZE)
                points.append((px, py))
            if len(points) >= 2:
                draw.line(points, fill=255, width=road_width)
        else:
            # Straight line between u and v
            node_u = G.nodes[u]
            node_v = G.nodes[v]
            px_u, py_u = latlon_to_pixel(node_u["y"], node_u["x"], bounds, OUTPUT_SIZE)
            px_v, py_v = latlon_to_pixel(node_v["y"], node_v["x"], bounds, OUTPUT_SIZE)
            draw.line([(px_u, py_u), (px_v, py_v)], fill=255, width=road_width)

    # 5. Save output
    stitched_img.save(img_name)
    mask_img.save(mask_name)
    print(f"  [+] Saved {prefix} tile: x={base_x}, y={base_y}")


def main():
    print("==========================================================")
    print("  ISRO NetSight - Data Generation Pipeline (OSM + ESRI)")
    print("  Enhanced: Road-type width calibration + CLAHE + 20 cities")
    print("==========================================================")

    samples_per_city = 8  # More samples for better diversity

    for city_name, (lat, lon) in CITIES.items():
        print(f"\nProcessing {city_name} (Lat: {lat}, Lon: {lon})...")

        # Get the central tile
        center_tile = mercantile.tile(lon, lat, ZOOM_LEVEL)

        # Sample tiles around the center in a grid
        count = 0
        for dx in range(-3, 4, 2):  # stride by 2 because we stitch 2x2
            for dy in range(-3, 4, 2):
                if count >= samples_per_city:
                    break
                generate_tile_pair(center_tile.x + dx, center_tile.y + dy, ZOOM_LEVEL, city_name)
                count += 1

    print("\nDataset generation complete!")
    print(f"Check {DATA_DIR}")


if __name__ == "__main__":
    main()
