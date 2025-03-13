import os
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm

# Paths
hr_input = r"D:\Super_Resolution\Delft\HR\hr_1km_1km\Delft_hr_1km_1km_2_6.tif"
lr_input = r"D:\Super_Resolution\Delft\LR\lr_1km_1km\Delft_lr_1km_1km_2_6.tif"
hr_upscaled_path = r"D:\Super_Resolution\Delft\HR\upscaled_hr_6cm.tif"
hr_output_dir = r"D:\Super_Resolution\Delft\HR\tiles_256x256"
lr_output_dir = r"D:\Super_Resolution\Delft\LR\tiles_64x64"

# Function to clear output directories
def clear_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(directory)

# Clear directories before processing
clear_directory(hr_output_dir)
clear_directory(lr_output_dir)

# **Step 1: Upscale HR Raster to 6.25cm Resolution**
with rasterio.open(hr_input) as hr_src:
    # Compute new dimensions
    scale_factor = 6.25 / 8.0  # Correct scaling factor from 8 cm to 6.25 cm
    new_width = int(hr_src.width / scale_factor)
    new_height = int(hr_src.height / scale_factor)

    # Correct transformation update
    new_transform = hr_src.transform * rasterio.Affine.scale(scale_factor, scale_factor)

    # Update profile
    hr_profile = hr_src.profile.copy()
    hr_profile.update({
        "width": new_width,
        "height": new_height,
        "transform": new_transform,
        "dtype": "uint8",
        "count": hr_src.count  # Ensure the same number of bands
    })

    # Read & Resample
    upscaled_hr = hr_src.read(
        out_shape=(hr_src.count, new_height, new_width),
        resampling=rasterio.enums.Resampling.bilinear  # Use bilinear to avoid artifacts
    )

    # Save upscaled HR raster
    with rasterio.open(hr_upscaled_path, "w", **hr_profile) as dst:
        dst.write(upscaled_hr)

# **Step 2: Export HR Tiles (256x256)**
hr_patch_size = 256  # HR tile size at 6.25cm resolution
hr_overlap_ratio = 0.25  # 25% overlap
hr_stride = int(hr_patch_size * (1 - hr_overlap_ratio))  # Compute stride in pixels

with rasterio.open(hr_upscaled_path) as hr_src:
    width, height = hr_src.width, hr_src.height
    hr_transform = hr_src.transform
    tile_count = 0

    for y in tqdm(range(0, height - hr_patch_size + 1, hr_stride)):
        for x in range(0, width - hr_patch_size + 1, hr_stride):
            # Define HR tile window
            hr_window = Window(x, y, hr_patch_size, hr_patch_size)
            hr_tile = hr_src.read(window=hr_window)

            # Use geospatial coordinates to name HR tiles
            tile_bounds = rasterio.windows.bounds(hr_window, hr_transform)
            tile_filename = os.path.join(hr_output_dir, f"HR_{tile_count}_{int(tile_bounds[0])}_{int(tile_bounds[1])}.tif")

            # Save HR tile with correct georeferencing
            hr_profile.update({
                "width": hr_patch_size,
                "height": hr_patch_size,
                "transform": rasterio.windows.transform(hr_window, hr_transform)
            })

            with rasterio.open(tile_filename, "w", **hr_profile) as dst:
                dst.write(hr_tile)

            tile_count += 1

print(f"HR Tiling Completed: {tile_count} tiles saved.")

# **Step 3: Export LR Tiles (64x64)**
lr_tile_size = 64  # LR tile size at 25cm resolution
lr_overlap_ratio = 0.25  # 25% overlap
lr_stride = int(lr_tile_size * (1 - lr_overlap_ratio))  # Compute stride in pixels

with rasterio.open(lr_input) as lr_src:
    width, height = lr_src.width, lr_src.height
    lr_transform = lr_src.transform
    tile_count = 0

    for y in tqdm(range(0, height - lr_tile_size + 1, lr_stride)):
        for x in range(0, width - lr_tile_size + 1, lr_stride):
            # Define LR tile window
            lr_window = Window(x, y, lr_tile_size, lr_tile_size)
            lr_tile = lr_src.read(window=lr_window)

            # Use geospatial coordinates to name LR tiles
            tile_bounds = rasterio.windows.bounds(lr_window, lr_transform)
            tile_filename = os.path.join(lr_output_dir, f"LR_{tile_count}_{int(tile_bounds[0])}_{int(tile_bounds[1])}.tif")

            # Save LR tile with correct georeferencing
            lr_profile = lr_src.profile.copy()
            lr_profile.update({
                "width": lr_tile_size,
                "height": lr_tile_size,
                "transform": rasterio.windows.transform(lr_window, lr_transform)
            })

            with rasterio.open(tile_filename, "w", **lr_profile) as dst:
                dst.write(lr_tile)

            tile_count += 1

print(f"LR Tiling Completed: {tile_count} tiles saved.")
