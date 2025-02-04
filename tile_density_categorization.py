import geopandas as gpd
import rasterio
from shapely.geometry import box
from tqdm import tqdm
import shutil
import re
from shapely.validation import make_valid
import os

# Paths
tiles_dir = r"D:\Super_Resolution\Rotterdam\real_hr\tiles_256_1_1"  # Tiles folder
land_use_path = r"D:\Super_Resolution\shp\land_use_rott.shp"  # Land use shapefile

# Define the grid size based on raster pixel size
grid_size = 0.08  # Ensure this matches the raster's pixel resolution

# Load land use data
land_use = gpd.read_file(land_use_path)
land_use = land_use[["geometry", "class_2018"]]  # Keep only relevant columns

# Strip spaces and ensure all category names are clean
land_use["class_2018"] = land_use["class_2018"].str.strip()

# Debugging: Print all unique land-use categories before any modifications
print("Unique land-use categories before processing:")
print(land_use["class_2018"].unique())

# Ensure all geometries are valid
invalid_geometries = land_use[~land_use.is_valid]
if not invalid_geometries.empty:
    print(f"Fixing {len(invalid_geometries)} invalid geometries...")
    land_use["geometry"] = land_use["geometry"].apply(make_valid)

# Ensure multipolygons are properly handled by converting them into individual polygons
land_use = land_use.explode(index_parts=False)
# Normalize category names before mapping
land_use["class_2018"] = land_use["class_2018"].str.replace(r"\s*-\s*", "-", regex=True).str.strip()

category_mapping = {
    "Continuous urban fabric (S.L.:>80%)": "High-Density Urban",
    "Discontinuous dense urban fabric (S.L.: 50%-80%)": "High-Density Urban",
    "Discontinuous medium density urban fabric (S.L.: 30%-50%)": "Low-Density Urban",
    "Discontinuous low density urban fabric (S.L.: 10%-30%)": "Low-Density Urban",
    "Discontinuous very low density urban fabric (S.L.:<10%)": "Low-Density Urban",
    "Isolated structures": "Low-Density Urban",
    "Water": "Non-Urban / Green",
    "Sports and leisure facilities": "Non-Urban / Green",
    "Land without current use": "Non-Urban / Green",
    "Forests": "Non-Urban / Green",
    "Green urban areas": "Non-Urban / Green",
    "Pastures": "Non-Urban / Green",
    "Herbaceous vegetation associations (natural grassland, moors...)": "Non-Urban / Green",
    "Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)": "Non-Urban / Green",
    "Wetlands": "Non-Urban / Green",
    "Arable land (annual crops)": "Non-Urban / Green",
    "Industrial, commercial, public, military and private units": "Industrial & Infrastructure",
    "Port areas": "Industrial & Infrastructure",
    "Railways and associated land": "Industrial & Infrastructure",
    "Mineral extraction and dump sites": "Industrial & Infrastructure",
    "Other roads and associated land": "Industrial & Infrastructure",
    "Construction sites": "Industrial & Infrastructure",
    "Fast transit roads and associated land": "Industrial & Infrastructure",
    "Airports": "Industrial & Infrastructure",
    "Permanent crops (vineyards, fruit trees, olive groves)": "Non-Urban / Green",
}


# Standardize category names
land_use["class_2018"] = (
    land_use["class_2018"]
    .str.replace(r"\s*:\s*", ": ", regex=True)  # Standardize spaces around colons
    .str.replace(r"\s*-\s*", "-", regex=True)  # Standardize spaces around dashes
    .str.replace(r"\s*>\s*", ">", regex=True)  # Remove extra spaces around ">"
    .str.replace(r"\s*<\s*", "<", regex=True)  # Remove extra spaces around "<"
    .str.strip()  # Remove leading/trailing spaces
)


print("\nDebugging: Exact Land-Use Categories Before Mapping:")
for category in land_use["class_2018"].unique():
    print(f"'{category}'")

# Apply mapping
land_use["merged_category"] = land_use["class_2018"].map(category_mapping)



# Debugging: Print categories that were NOT mapped
unmapped_categories = land_use[land_use["merged_category"].isna()]["class_2018"].unique()
if len(unmapped_categories) > 0:
    print("\n⚠ WARNING: These land-use categories were NOT mapped:")
    print(unmapped_categories)

# Debugging: Check if any categories were dropped during mapping
print("\nUnique categories after mapping:")
print(land_use["merged_category"].unique())

# Remove NaN values (categories that were not mapped)
land_use = land_use.dropna(subset=["merged_category"])

# Debugging: Verify final categories before processing
print("\nFinal land-use categories before intersection:")
print(land_use["merged_category"].unique())

# Function to clean category names for folder names
def clean_name(name):
    return re.sub(r'[<>:"/\\|?*]', '', name).replace(" ", "_")

# Create category folders
unique_classes = land_use["merged_category"].unique()
output_dirs = {cls: os.path.join(tiles_dir, clean_name(cls)) for cls in unique_classes}

for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)

# Function to snap values to a grid size
def snap_to_grid(value, grid_size):
    return round(value / grid_size) * grid_size

# Process each tile
tiles = [f for f in os.listdir(tiles_dir) if f.endswith(".tif")]
print(f"\nProcessing {len(tiles)} tiles...")

for tile in tqdm(tiles, desc="Categorizing Tiles"):
    tile_path = os.path.join(tiles_dir, tile)

    with rasterio.open(tile_path) as src:
        bounds = src.bounds

        # Snap tile bounds to the exact raster grid size
        x_min = snap_to_grid(bounds.left, grid_size)
        x_max = snap_to_grid(bounds.right, grid_size)
        y_min = snap_to_grid(bounds.bottom, grid_size)
        y_max = snap_to_grid(bounds.top, grid_size)

        # Create grid-aligned tile boundary
        tile_bounds = box(x_min, y_min, x_max, y_max)
        tile_gdf = gpd.GeoDataFrame({"geometry": [tile_bounds]}, crs=src.crs)

    tile_gdf = tile_gdf.to_crs(land_use.crs)

    # Debugging: Print tile bounds
    print(f"\nTile {tile} Bounds: {tile_bounds.bounds}")

    # Find all intersections
    intersecting_land_use = gpd.overlay(land_use, tile_gdf, how="intersection")

    if not intersecting_land_use.empty:
        # Calculate the intersection area for each category
        intersecting_land_use["area"] = intersecting_land_use.geometry.area

        # Debugging: Print categories intersecting the tile
        print(f"Intersecting categories for {tile}:")
        print(intersecting_land_use[["merged_category", "area"]])

        dominant_class = intersecting_land_use.loc[intersecting_land_use["area"].idxmax(), "merged_category"]

        clean_class = clean_name(dominant_class)
        shutil.copy(tile_path, os.path.join(output_dirs[dominant_class], tile))

print("Categorization complete. Check the categorized tile folders.")
