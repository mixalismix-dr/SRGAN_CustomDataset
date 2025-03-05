import os
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm

# Input and output paths
input_file = r"D:\Super_Resolution\Rotterdam\hr_1km_1km\Rotterdam_hr_1km_1km_1_1.tif"
output_dir = r"D:\Super_Resolution\Rotterdam\real_hr\tiles_256_1_1"

# Tile size and overlap
tile_size = 256
overlap_ratio = 0.10  # 10% overlap
overlap = int(tile_size * overlap_ratio)

# Create output directory if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)

# Open raster file
with rasterio.open(input_file) as src:
    width, height = src.width, src.height

    # Compute step size (stride)
    stride = tile_size - overlap

    # Generate tiles
    tile_count = 0
    for y in tqdm(range(0, height, stride), desc="Processing Rows"):
        for x in tqdm(range(0, width, stride), desc="Processing Columns", leave=False):
            if x + tile_size > width or y + tile_size > height:
                continue  # Skip incomplete tiles

            window = Window(x, y, tile_size, tile_size)
            tile = src.read(window=window)

            # Save tile
            tile_filename = f"{output_dir}/tile_{x}_{y}.tif"
            profile = src.profile
            profile.update({
                "width": tile_size,
                "height": tile_size,
                "transform": rasterio.windows.transform(window, src.transform)
            })

            with rasterio.open(tile_filename, "w", **profile) as dst:
                dst.write(tile)

            tile_count += 1

print(f"Tiling completed. {tile_count} tiles saved in '{output_dir}'.")
