import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.warp import reproject

# Input raster (8 cm resolution)
input_raster = r"D:\Super_Resolution\Utrecht\utrecht_hr.tif"
output_tile = r"D:\Super_Resolution\Utrecht\utrecht_hr_6cm.tif"

# Parameters
tile_size_m = 1000            # 1km
original_res = 0.08           # 8 cm
upsampled_res = 0.0625         # 6 cm

tile_size_px = int(tile_size_m / original_res)  # = 12500

# Choose origin of tile in pixel space
tile_x = 0  # top-left x in pixel coords
tile_y = 0  # top-left y in pixel coords

with rasterio.open(input_raster) as src:
    window = Window(tile_x, tile_y, tile_size_px, tile_size_px)
    transform = src.window_transform(window)
    tile = src.read(window=window)
    count, height, width = tile.shape

    new_height = int(height * original_res / upsampled_res)
    new_width = int(width * original_res / upsampled_res)

    new_transform = transform * transform.scale(width / new_width, height / new_height)

    profile = src.profile.copy()
    profile.update({
        "height": new_height,
        "width": new_width,
        "transform": new_transform,
        "dtype": "uint8",
        "count": count,
    })

    with rasterio.open(output_tile, "w", **profile) as dst:
        for i in range(count):
            reproject(
                source=tile[i],
                destination=rasterio.band(dst, i + 1),
                src_transform=transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=Resampling.cubic,
            )

print("Saved:", output_tile)
