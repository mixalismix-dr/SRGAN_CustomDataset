import geopandas as gpd
import rasterio
from shapely.geometry import box
from tqdm import tqdm
import shutil
import os
import re

##For the cleaned shps

# Paths
tiles_dir = r"D:\Super_Resolution\Zwolle\Iteration_2\LR_tiles_4_3_64x64"  # Tiles folder
land_use_path = r"D:\Super_Resolution\data\land_use\land_use_zwolle.shp"  # Already cleaned shapefile

# Grid size
grid_size = 0.25

# Load land use
land_use = gpd.read_file(land_use_path)
land_use = land_use[["geometry", "merged_cat"]]
land_use = land_use.explode(index_parts=False)

# Ensure valid geometries
land_use["geometry"] = land_use["geometry"].buffer(0)

# Prepare output folders
def clean_name(name):
    return re.sub(r'[<>:"/\\|?*]', '', name).replace(" ", "_")

unique_classes = land_use["merged_cat"].unique()
output_dirs = {cls: os.path.join(tiles_dir, clean_name(cls)) for cls in unique_classes}
for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)

# Snap helper
def snap_to_grid(value, grid_size):
    return round(value / grid_size) * grid_size

# Process tiles
tiles = [f for f in os.listdir(tiles_dir) if f.endswith(".tif")]
print(f"\nProcessing {len(tiles)} tiles...")

for tile in tqdm(tiles, desc="Categorizing Tiles"):
    tile_path = os.path.join(tiles_dir, tile)

    with rasterio.open(tile_path) as src:
        bounds = src.bounds
        x_min = snap_to_grid(bounds.left, grid_size)
        x_max = snap_to_grid(bounds.right, grid_size)
        y_min = snap_to_grid(bounds.bottom, grid_size)
        y_max = snap_to_grid(bounds.top, grid_size)
        tile_bounds = box(x_min, y_min, x_max, y_max)
        tile_gdf = gpd.GeoDataFrame({"geometry": [tile_bounds]}, crs=src.crs).to_crs(land_use.crs)

    intersecting = gpd.overlay(land_use, tile_gdf, how="intersection")

    if not intersecting.empty:
        intersecting["area"] = intersecting.geometry.area
        dominant = intersecting.loc[intersecting["area"].idxmax(), "merged_cat"]
        shutil.copy(tile_path, os.path.join(output_dirs[dominant], tile))

print("\nCategorization complete.")
