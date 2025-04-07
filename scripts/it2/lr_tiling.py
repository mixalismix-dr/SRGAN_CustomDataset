import os
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm

# Paths
lr_input = r"D:\Super_Resolution\data\Zwolle_lr_1km_1km\Zwolle_lr_1km_1km_4_3.tif"
lr_output_dir = r"D:\Super_Resolution\Zwolle\Iteration_2\LR_tiles_4_3_64x64"
os.makedirs(lr_output_dir, exist_ok=True)

# Tile size & overlap
tile_size = 64
overlap_ratio = 0.25  # 25% overlap
overlap = int(tile_size * overlap_ratio)
stride = tile_size - overlap  # Stride ensures proper overlap

# **Clear directory before processing**
def clear_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(directory)

clear_directory(lr_output_dir)

# **Read LR Raster and Extract 64x64 Tiles**
with rasterio.open(lr_input) as src:
    width, height = src.width, src.height
    lr_transform = src.transform
    tile_count = 0

    for y in tqdm(range(0, height - tile_size + 1, stride), desc="Exporting LR Tiles"):
        for x in range(0, width - tile_size + 1, stride):
            # Define window for cropping
            window = Window(x, y, tile_size, tile_size)
            tile = src.read(window=window)

            # Compute georeferencing transform for the tile
            tile_transform = rasterio.windows.transform(window, lr_transform)

            # Define output file name
            tile_filename = os.path.join(lr_output_dir, f"LR_tile_{tile_count}.tif")

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

print(f"LR Tiling Completed: {tile_count} tiles saved.")
