import os
from osgeo import gdal

# Input and output paths
input_file = r"D:\Super_Resolution\Delft\LR\vrt\Delft_lr.vrt"  # Path to your VRT or raster file
output_dir = r"D:\Super_Resolution\Delft\LR\vrt\tiles_64x64"  # Directory to save tiles
metadata_dir = os.path.join(output_dir, "metadata")  # Directory to save metadata files
os.makedirs(metadata_dir, exist_ok=True)

# Tile size and overlap
tile_size = 64
overlap = 26
step_size = tile_size - overlap

# Open the input raster
dataset = gdal.Open(input_file)
if not dataset:
    raise FileNotFoundError(f"Unable to open the file: {input_file}")

raster_width = dataset.RasterXSize
raster_height = dataset.RasterYSize
geo_transform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

# Loop through the raster to check tiles
tile_count = 0
for x_off in range(0, raster_width - tile_size + 1, step_size):
    for y_off in range(0, raster_height - tile_size + 1, step_size):
        # Define output file name for the tile
        output_tile = os.path.join(output_dir, f"tile_{x_off}_{y_off}.tif")
        metadata_file = os.path.join(metadata_dir, f"tile_{x_off}_{y_off}.txt")

        # Check if the tile exists
        if os.path.exists(output_tile):
            # Calculate georeferencing information for the tile
            tile_geo_transform = (
                geo_transform[0] + x_off * geo_transform[1],  # Top-left x
                geo_transform[1],                            # Pixel width
                geo_transform[2],                            # Row rotation
                geo_transform[3] + y_off * geo_transform[5], # Top-left y
                geo_transform[4],                            # Column rotation
                geo_transform[5]                             # Pixel height
            )

            # Write metadata to a file
            with open(metadata_file, "w") as meta:
                meta.write(f"Tile Name: {os.path.basename(output_tile)}\n")
                meta.write(f"X Offset: {x_off}\n")
                meta.write(f"Y Offset: {y_off}\n")
                meta.write(f"Width: {tile_size} pixels\n")
                meta.write(f"Height: {tile_size} pixels\n")
                meta.write(f"GeoTransform: {tile_geo_transform}\n")
                meta.write(f"Projection: {projection}\n")
            print(f"Metadata saved: {metadata_file}")
        else:
            print(f"Tile does not exist: {output_tile}")

        tile_count += 1

print(f"Total tiles processed: {tile_count}")
print("Metadata generation completed!")
