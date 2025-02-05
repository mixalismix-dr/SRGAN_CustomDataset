import os
import geopandas as gpd
import rasterio
from shapely.geometry import box

# Define the root directory where categorized tile folders are stored
root_dir = r"D:\Super_Resolution\Rotterdam\real_hr\tiles_256"

# Output shapefile path
output_shp = os.path.join(root_dir, "tiles_extent.shp")

# Initialize an empty GeoDataFrame
gdf_list = []

# Iterate through each category folder
for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)

    # Ensure it's a directory
    if os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            if filename.endswith(".tif"):  # Process only TIFF files
                tile_path = os.path.join(category_path, filename)

                # Open raster to get bounding box
                with rasterio.open(tile_path) as src:
                    bounds = src.bounds
                    geometry = box(*bounds)  # Create a rectangle from bounds

                    # Append data to the list
                    gdf_list.append({
                        "filename": filename,
                        "category": category,
                        "geometry": geometry
                    })

# Convert list to GeoDataFrame
gdf = gpd.GeoDataFrame(gdf_list, crs="EPSG:28992")  # Set the correct CRS

# Save to a shapefile
gdf.to_file(output_shp)

print(f"Shapefile saved: {output_shp}")
