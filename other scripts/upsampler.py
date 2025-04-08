
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
import rasterio
from rasterio.transform import Affine
from PIL import Image
import numpy as np

input_image = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\test_data\p3_othercities\LR_tile_3613.tif"
output_image = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\test_data\p3_othercities\LR_tile_resampled_3613.tif"

target_res = 0.08  # 8 cm resolution

with rasterio.open(input_image) as src:
    data = src.read()  # (bands, H, W)
    profile = src.profile
    height, width = data.shape[1], data.shape[2]

    # Compute scale factor based on original resolution
    scale_factor = src.transform.a / target_res  # e.g. 0.25 / 0.08 = 3.125
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    upsampled = []
    for band in data:
        img_pil = Image.fromarray(band.astype(np.uint8))
        img_resized = img_pil.resize((new_width, new_height), Image.BICUBIC)
        upsampled.append(np.array(img_resized))

    upsampled_np = np.stack(upsampled, axis=0)

    # Adjust transform to reflect higher resolution
    new_transform = src.transform * Affine.scale(1 / scale_factor)

    profile.update({
        "height": new_height,
        "width": new_width,
        "transform": new_transform,
        "dtype": upsampled_np.dtype,
        "count": upsampled_np.shape[0]
    })

    with rasterio.open(output_image, "w", **profile) as dst:
        dst.write(upsampled_np)

print(f"Resampled raster saved to: {output_image}")

