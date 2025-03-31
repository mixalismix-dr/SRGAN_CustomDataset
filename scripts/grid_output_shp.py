import os
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Input and output paths
input_file = r"D:\Super_Resolution\Delft\HR\hr_1km_1km\Delft_hr_1km_1km_2_6.tif"
output_shapefile = r"D:\Super_Resolution\Delft\HR\Delft_grid_2_6.tif"

# Tile size and overlap
tile_size = 256
overlap_ratio = 0.10  # 10% overlap
overlap = int(tile_size * overlap_ratio)

# Open raster file and compute the virtual tile grid
with rasterio.open(input_file) as src:
    width, height = src.width, src.height
    transform = src.transform
    crs = src.crs
    stride = tile_size - overlap

    polygons = []
    ids = []

    for y in tqdm(range(0, height, stride), desc="Rows"):
        for x in range(0, width, stride):
            if x + tile_size > width or y + tile_size > height:
                continue

            window = Window(x, y, tile_size, tile_size)
            bounds = rasterio.windows.bounds(window, transform)
            geom = box(*bounds)
            polygons.append(geom)
            ids.append((x, y))

# Save to shapefile
gdf = gpd.GeoDataFrame({"geometry": polygons, "x": [i[0] for i in ids], "y": [i[1] for i in ids]}, crs=crs)
gdf.to_file(output_shapefile)

print(f"Grid shapefile saved to '{output_shapefile}'. Total tiles: {len(gdf)}")
