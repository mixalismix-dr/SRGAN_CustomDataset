
import rasterio
from rasterio.transform import Affine
from PIL import Image
import numpy as np


# # Input and output paths
# input_raster = r"D:\Super_Resolution\Rotterdam\hr_1km_1km\Rotterdam_hr_1km_1km_2_3.tif"
# output_raster = r"D:\Super_Resolution\Rotterdam\hr_1km_1km\Rotterdam_hr_1km_1km_2_3_6cm.tif"
#
# # Scale factor: (8cm → 6cm)
# scale_factor = 1.28
#
# # Open raster using rasterio
# with rasterio.open(input_raster) as src:
#     data = src.read()  # Read all bands
#     num_bands, height, width = data.shape
#
#     # Resize each band separately using PIL
#     upsampled_bands = []
#     for i in range(num_bands):
#         img = Image.fromarray(data[i])  # Convert each band to PIL image
#         new_size = (int(width * scale_factor), int(height * scale_factor))
#         upsampled_img = img.resize(new_size, Image.BICUBIC)
#         upsampled_bands.append(np.array(upsampled_img))
#
#     # Convert back to NumPy array
#     upsampled_data = np.stack(upsampled_bands, axis=0)
#
#     # Update metadata
#     transform = src.transform * src.transform.scale(1 / scale_factor, 1 / scale_factor)
#     new_meta = src.meta.copy()
#     new_meta.update({
#         "height": new_size[1],
#         "width": new_size[0],
#         "transform": transform
#     })
#
#     # Write the upsampled raster
#     with rasterio.open(output_raster, "w", **new_meta) as dst:
#         dst.write(upsampled_data)
#
# print(f"Upsampled raster saved as {output_raster}")

input_image = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\other scripts\LR_tile_569.tif"
output_image = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\other scripts\resampled_569.tif"

target_res = 0.08  # 8 cm resolution
target_size = (256, 256)

with rasterio.open(input_image) as src:
    img = src.read(1)  # Read first band (grayscale)
    profile = src.profile  # Get metadata

    # Convert to PIL Image (ensure grayscale mode 'L')
    img_pil = Image.fromarray(img).convert("L")
    img_resampled = img_pil.resize(target_size, Image.BICUBIC)

    # Convert back to NumPy array with original dtype
    img_resampled_np = np.array(img_resampled, dtype=img.dtype)

    # Compute new affine transformation (upper-left remains fixed)
    scale_x = target_res / src.transform.a
    scale_y = target_res / -src.transform.e  # Negative because Y is flipped
    new_transform = src.transform * Affine.scale(scale_x, scale_y)

    # Update profile
    profile.update({
        "height": target_size[1],
        "width": target_size[0],
        "transform": new_transform,
        "dtype": img_resampled_np.dtype,
        "count": 1  # Ensure output is single-band grayscale
    })

    # Write output raster
    with rasterio.open(output_image, "w", **profile) as dst:
        dst.write(img_resampled_np, 1)

print(f"Resampled grayscale image saved to: {output_image}")

