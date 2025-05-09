import os
import re
from multiprocessing import Pool
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.windows import from_bounds
from shapely.geometry import box
from tqdm import tqdm

# Input Paths
land_use_path = r"/mnt/SRGAN/data/land_use_sgravenhage_2.geojson"
grid_path = r"/mnt/SRGAN/data/Delft_grid.geojson"
raster_path = r"/mnt/SRGAN/data/delft_lr.tif"
output_base = r"/mnt/SRGAN/data"

def clean_name(name):
    return re.sub(r'[<>:"/\\|?*]', '', name).replace(" ", "_")

def create_output_dirs(categories):
    for category in categories:
        os.makedirs(os.path.join(output_base, clean_name(category)), exist_ok=True)

def has_any_zeros(tile_data):
    if tile_data.size == 0:
        return True
    return np.any(tile_data == 0)

def process_tile(args):
    row, land_use, raster_path, land_use_spatial_index = args
    tile_geom = row.geometry
    tile_bounds = tile_geom.bounds

    possible_matches_index = list(land_use_spatial_index.intersection(tile_bounds))
    if not possible_matches_index:
        return None

    possible_matches = land_use.iloc[possible_matches_index]
    intersecting = gpd.overlay(
        possible_matches,
        gpd.GeoDataFrame(geometry=[tile_geom], crs=land_use.crs),
        how="intersection"
    )

    if intersecting.empty:
        return None

    intersecting["area"] = intersecting.geometry.area
    dominant_category = intersecting.loc[intersecting["area"].idxmax(), "merged_cat"]

    with rasterio.open(raster_path) as src:
        try:
            window = from_bounds(*tile_bounds, transform=src.transform)
            if window.height <= 0 or window.width <= 0:
                return None

            transform = src.window_transform(window)
            tile_data = src.read(window=window)

            if has_any_zeros(tile_data):
                return None

            profile = src.profile
            profile.update({
                "height": window.height,
                "width": window.width,
                "transform": transform
            })

            tile_filename = f"tile_{row.name:05d}.tif"
            output_dir = os.path.join(output_base, clean_name(dominant_category))
            with rasterio.open(os.path.join(output_dir, tile_filename), "w", **profile) as dst:
                dst.write(tile_data)

            return row.name
        except Exception as e:
            print(f"Error processing tile {row.name}: {str(e)}")
            return None

def main():
    print("Loading grid...")
    grid = gpd.read_file(grid_path)

    print("Loading land use data...")
    land_use = gpd.read_file(land_use_path)[["geometry", "merged_cat"]]
    land_use = land_use.to_crs(grid.crs)

    print("Creating spatial index...")
    land_use_spatial_index = land_use.sindex

    unique_categories = land_use["merged_cat"].unique()
    create_output_dirs(unique_categories)

    args = [(row, land_use, raster_path, land_use_spatial_index) for _, row in grid.iterrows()]

    print("Processing tiles...")
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_tile, args), total=len(args)))

    successful = sum(1 for r in results if r is not None)
    print(f"\nProcessing complete. Successfully exported {successful} of {len(grid)} tiles.")

if __name__ == "__main__":
    main()
