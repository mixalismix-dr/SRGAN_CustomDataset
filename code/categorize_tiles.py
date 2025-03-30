import os
import random
import rasterio
import shutil
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from tqdm import tqdm
from typing import List


def generate_virtual_grid(raster_path, tile_size, overlap_ratio):
    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height

        pixel_size = abs(transform[0])
        stride = tile_size - int(tile_size * overlap_ratio)
        grid = []
        tile_indices = []

        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                x_min, y_max = transform * (x, y)
                x_max, y_min = transform * (x + tile_size, y + tile_size)
                geom = box(x_min, y_min, x_max, y_max)
                grid.append(geom)
                tile_indices.append((x, y))

        gdf = gpd.GeoDataFrame({"geometry": grid, "index_xy": tile_indices}, crs=crs)
        return gdf


def export_tiles(raster_path, tiles_gdf, tile_size, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(raster_path) as src:
        for i, row in tiles_gdf.iterrows():
            x, y = row["index_xy"]
            window = rasterio.windows.Window(x, y, tile_size, tile_size)
            transform = rasterio.windows.transform(window, src.transform)
            tile = src.read(window=window)
            profile = src.profile.copy()
            profile.update({
                "height": tile_size,
                "width": tile_size,
                "transform": transform
            })
            filename = f"{prefix}_tile_{i:05d}_{row['city']}.tif"
            output_path = os.path.join(output_dir, filename)
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(tile)


def categorize_tiles(hr_raster, lr_raster, land_use_path, city, output_hr_dir, output_lr_dir, total_samples: int, all_cities: List[str], iteration: int):
    print(f"\nGenerating virtual grid for {city} (Iteration {iteration})")

    tile_size_hr = 256
    tile_size_lr = 64
    overlap_hr = 0.25 if iteration == 2 else 0.10
    overlap_lr = 0.25

    virtual_grid = generate_virtual_grid(hr_raster, tile_size_hr, overlap_hr)
    virtual_grid["city"] = city

    land_use = gpd.read_file(land_use_path)[["geometry", "merged_cat"]]
    land_use = land_use.to_crs(virtual_grid.crs)

    print(f"Overlaying grid with land use for {city}")
    joined = gpd.overlay(virtual_grid, land_use, how="intersection")
    joined["area"] = joined.geometry.area
    joined = joined.sort_values("area", ascending=False).drop_duplicates("index_xy")

    weights = {
        "High-Density Urban": 2,
        "Low-Density Urban": 2,
        "Non-Urban / Green": 1,
        "Industrial & Infrastructure": 1
    }

    per_city_samples = total_samples // len(all_cities)
    total_weight = sum(weights.values())
    target_per_category = {cat: int((w / total_weight) * per_city_samples) for cat, w in weights.items()}

    selected = []

    for category, target_count in target_per_category.items():
        matches = joined[joined["merged_cat"] == category]
        if len(matches) >= target_count:
            selected_tiles = matches.sample(target_count, random_state=1)
        else:
            fallback_cat = "High-Density Urban" if category == "Low-Density Urban" else "Low-Density Urban"
            fallback = joined[joined["merged_cat"] == fallback_cat]
            fallback_needed = target_count - len(matches)
            extra = fallback.sample(min(fallback_needed, len(fallback)), random_state=1) if not fallback.empty else gpd.GeoDataFrame()
            selected_tiles = gpd.GeoDataFrame(pd.concat([matches, extra]))

        selected.append(selected_tiles)

    selected_gdf = gpd.GeoDataFrame(pd.concat(selected), crs=virtual_grid.crs)

    print(f"\nExporting {len(selected_gdf)} tiles for {city}")
    export_tiles(hr_raster, selected_gdf, tile_size_hr, output_hr_dir, prefix="HR")
    export_tiles(lr_raster, selected_gdf, tile_size_lr, output_lr_dir, prefix="LR")
