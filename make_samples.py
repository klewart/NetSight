import numpy as np
import json
import os
from PIL import Image

# 1. Create the dummy directories
os.makedirs("data/sample_images", exist_ok=True)
os.makedirs("data/sample_geojson", exist_ok=True)

# 2. Save a fake 3-band satellite image patch (512x512 pixels)
fake_satellite_pixels = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
img = Image.fromarray(fake_satellite_pixels)
img.save("data/sample_images/AOI_2_Vegas_Scene_001.tif")

# 3. Save a fake matching road GeoJSON entry
fake_geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [100.0, 100.0],
                    [200.0, 300.0],
                    [400.0, 450.0]
                ]
            },
            "properties": {"road_type": "highway"}
        }
    ]
}

with open("data/sample_geojson/AOI_2_Vegas_Scene_001.geojson", "w") as f:
    json.dump(fake_geojson_data, f)

print("✅ Success: Synthetic sample image and geojson matching pairs generated inside /data!")