import os
import cv2
import numpy as np
from PIL import Image
import rasterio

tile_dir = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\custom_dataset\train_HR3"
mask_dir = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\other scripts\output_mask_tiles"
output_dir = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\custom_dataset\train_HR3_rgba"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(tile_dir):
    if filename.endswith(".tif"):
        tile_path = os.path.join(tile_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {filename}, skipping.")
            continue

        tile = cv2.imread(tile_path, cv2.IMREAD_COLOR)
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(mask_path).convert("L")).astype(np.uint8)

        if mask.shape[:2] != tile.shape[:2]:
            mask = cv2.resize(mask, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_NEAREST)

        with rasterio.open(tile_path) as src:
            transform = src.transform
            crs = src.crs

        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=tile.shape[0], width=tile.shape[1],
            count=4, dtype='uint8', crs=crs, transform=transform
        ) as dst:
            dst.write(tile[:, :, 0], 1)
            dst.write(tile[:, :, 1], 2)
            dst.write(tile[:, :, 2], 3)
            dst.write(mask, 4)

        print(f"Processed {filename}")
