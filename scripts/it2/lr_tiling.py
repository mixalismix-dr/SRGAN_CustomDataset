import os
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import cv2


lr_input = r"D:\Super_Resolution\Delft\LR\lr_1km_1km\Delft_lr_1km_1km_2_6.tif"
lr_output_dir = r"D:\Super_Resolution\Delft\LR\real_lr\tiles_64x64"
os.makedirs(lr_output_dir, exist_ok=True)

tile_size = 64
overlap_ratio = 0.10
overlap = int(tile_size * overlap_ratio)

with rasterio.open(lr_input) as src:
    width, height = src.width, src.height
    stride = tile_size - overlap
    tile_count = 0

    for y in tqdm(range(0, height, stride)):
        for x in range(0, width, stride):
            if x + tile_size > width or y + tile_size > height:
                continue

            window = Window(x, y, tile_size, tile_size)
            tile = src.read(window=window)

            tile_filename = os.path.join(lr_output_dir, f"LR_tile_{x}_{y}.tif")
            profile = src.profile.copy()
            profile.update({
                "width": tile_size,
                "height": tile_size,
                "transform": rasterio.windows.transform(window, src.transform)
            })

            with rasterio.open(tile_filename, "w", **profile) as dst:
                dst.write(tile)

            tile_count += 1

print(f"LR tiling completed. {tile_count} tiles saved.")
