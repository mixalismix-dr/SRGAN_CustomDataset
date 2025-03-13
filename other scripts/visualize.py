import os
import rasterio
import numpy as np
from PIL import Image
from rasterio.transform import Affine

# Define the folder path
folder_path = r"C:\Users\mike_\OneDrive\Desktop\vizualize"

# Get all .tif and .png files and sort them alphabetically
tif_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".tif")])
png_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

# Filter for LR and res files
lr_files = [f for f in tif_files if f.startswith("LR_tile_")]
res_files = [f for f in png_files if f.startswith("res_")]

# Ensure there's a one-to-one mapping
if len(lr_files) != len(res_files):
    print("Warning: Number of LR and res files do not match.")

# Process each pair
for lr_file, res_file in zip(lr_files, res_files):
    lr_path = os.path.join(folder_path, lr_file)
    res_path = os.path.join(folder_path, res_file)
    output_path = os.path.join(folder_path, f"{res_file[:-4]}_georef.tif")  # Append _georef

    try:
        # Read georeferencing from LR file
        with rasterio.open(lr_path) as src:
            transform = src.transform
            crs = src.crs
            profile = src.profile
            lr_width, lr_height = src.width, src.height

        # Open the PNG file to get new dimensions
        img = Image.open(res_path).convert("RGB")  # Ensure it's RGB
        res_width, res_height = img.size  # This should be 256x256

        # Compute scaling factor
        scale_x = lr_width / res_width  # Should be 64/256 = 0.25
        scale_y = lr_height / res_height  # Should be 64/256 = 0.25

        # Adjust the geotransform for the new resolution
        new_transform = Affine(
            transform.a * scale_x, transform.b, transform.c,
            transform.d, transform.e * scale_y, transform.f
        )

        # Update profile for 3-band RGB image
        profile.update(
            dtype=rasterio.uint8,
            count=3,  # Three channels for RGB
            transform=new_transform,
            crs=crs,
            width=res_width,
            height=res_height
        )

        # Save as GeoTIFF with updated georeferencing
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(3):  # Write RGB channels separately
                dst.write(np.array(img)[:, :, i], i + 1)

        print(f"Georeferencing applied: {output_path}")

    except Exception as e:
        print(f"Error processing {lr_file} -> {res_file}: {e}")
