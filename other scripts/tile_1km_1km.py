import os
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

# Paths
input_raster = r"D:\Super_Resolution\data\rasters\Zwolle_hr_6cm.tif"
output_dir = r"D:\Super_Resolution\data\Zwolle_hr_1km_1km"
tile_size_m = 1000  # in meters

# Create output folder
os.makedirs(output_dir, exist_ok=True)

with rasterio.open(input_raster) as src:
    transform = src.transform
    pixel_size = transform.a  # assumes square pixels (transform.a == transform.e)
    tile_size_px = int(tile_size_m / pixel_size)

    width = src.width
    height = src.height

    num_tiles_x = width // tile_size_px
    num_tiles_y = height // tile_size_px

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            window = Window(j * tile_size_px, i * tile_size_px, tile_size_px, tile_size_px)
            transform_window = src.window_transform(window)

            out_profile = src.profile.copy()
            out_profile.update({
                "height": tile_size_px,
                "width": tile_size_px,
                "transform": transform_window
            })

            out_path = os.path.join(output_dir, f"Zwolle_hr_1km_1km_{i}_{j}.tif")
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(src.read(window=window))

print(f"Done. Exported tiles to: {output_dir}")
