import os
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
from PIL import Image

# Paths
hr_upscaled_path = r"D:\Super_Resolution\Rotterdam\hr_1km_1km\Rotterdam_hr_1km_1km_2_3_6cm.tif"
hr_output_dir = r"D:\Super_Resolution\Rotterdam\HR\iteration_2\tiles_256x256"

# Ensure output directory exists
os.makedirs(hr_output_dir, exist_ok=True)

# Tile size & overlap
tile_size = 256
overlap_ratio = 0.25  # 25% overlap
overlap = int(tile_size * overlap_ratio)
stride = tile_size - overlap  # Calculate stride

# Clear output directory before processing
def clear_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(directory)

clear_directory(hr_output_dir)

# **Read HR Upscaled Raster and Extract 256x256 Tiles**
with rasterio.open(hr_upscaled_path) as src:
    width, height = src.width, src.height
    hr_transform = src.transform
    tile_count = 0

    for y in tqdm(range(0, height - tile_size + 1, stride), desc="Exporting HR Tiles"):
        for x in range(0, width - tile_size + 1, stride):
            # Define window for cropping
            window = Window(x, y, tile_size, tile_size)
            tile = src.read(window=window)

            # Compute georeferencing transform for the tile
            tile_transform = rasterio.windows.transform(window, hr_transform)

            # Define output file name using geospatial position
            tile_filename = os.path.join(hr_output_dir, f"HR_tile_{tile_count}.tif")

            # Copy profile & update for tile size
            profile = src.profile.copy()
            profile.update({
                "width": tile_size,
                "height": tile_size,
                "transform": tile_transform
            })

            # Save the tile
            with rasterio.open(tile_filename, "w", **profile) as dst:
                dst.write(tile)

            tile_count += 1

print(f"HR Tiling Completed: {tile_count} tiles saved.")
