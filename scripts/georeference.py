import os
import glob
import rasterio
from rasterio.transform import Affine
import numpy as np

# Paths
sr_root = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\result\delft3"
hr_root = r"D:\Super_Resolution\Delft\HR\real_hr\tiles_256"
up_root = r"D:\Super_Resolution\Delft\HR\generated_hr_normal_upscale\tiles_256"

# Function to find all TIFF images inside subdirectories
def get_all_images(root_dir):
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.tif"), recursive=True))

# Collect images from all subdirectories
hr_images = get_all_images(hr_root)
sr_images = get_all_images(sr_root)
up_images = get_all_images(up_root)

# Ensure HR images are used as reference
for hr_img in hr_images:
    tile_name = os.path.basename(hr_img).replace(".tif", "")

    # Find corresponding SR and UP images
    sr_candidates = [img for img in sr_images if f"res_{tile_name}" in img]
    up_candidates = [img for img in up_images if f"{tile_name}_lanczos_down_lanczos_up" in img]

    if not sr_candidates or not up_candidates:
        print(f"Skipping {tile_name}: SR or UP image missing")
        continue

    sr_img = sr_candidates[0]
    up_img = up_candidates[0]

    # Read CRS & Transform from HR image
    with rasterio.open(hr_img) as hr_ds:
        crs = hr_ds.crs
        transform = hr_ds.transform
        profile = hr_ds.profile

    # Function to apply georeferencing
    def reproject_and_save(image_path, output_path):
        with rasterio.open(image_path) as src:
            data = src.read()
            profile.update({
                "transform": transform,
                "crs": crs,
                "height": data.shape[1],
                "width": data.shape[2],
                "dtype": "uint8",
            })
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data)

    # Apply georeferencing
    reproject_and_save(sr_img, sr_img.replace(".tif", "_georef.tif"))
    reproject_and_save(up_img, up_img.replace(".tif", "_georef.tif"))

    print(f"Updated georeferencing for: {tile_name}")

