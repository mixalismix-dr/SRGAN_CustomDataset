import os
import rasterio
from PIL import Image
import numpy as np
from tqdm import tqdm


def crop_mask_based_on_tiles(tiles_dir, mask_path, output_mask_path, tile_size=256, overlap_ratio=0.10):
    # Ensure the output directory exists
    os.makedirs(output_mask_path, exist_ok=True)

    # Open the mask file using rasterio
    with rasterio.open(mask_path) as mask_img:
        mask_array = mask_img.read(1)  # Read the first band (binary mask)
        transform = mask_img.transform  # Georeferencing information
        crs = mask_img.crs
        dtype = mask_img.dtypes[0]

        mask_height, mask_width = mask_array.shape

        # Calculate the stride for tiling (this depends on your overlap)
        overlap = int(tile_size * overlap_ratio)
        stride = tile_size - overlap

        # Iterate through the tiles folder
        tile_count = 0
        for tile_name in tqdm(os.listdir(tiles_dir), desc="Processing Tiles"):
            # Get the coordinates from the tile name
            tile_coordinates = tile_name.replace('res_tile_', '').replace('.tif', '').split('_')
            tile_x, tile_y = int(tile_coordinates[0]), int(tile_coordinates[1])

            # Calculate the bounding box of the tile in the mask
            mask_tile = mask_array[tile_y:tile_y + tile_size, tile_x:tile_x + tile_size]

            # Define the tile's georeference based on its position
            tile_transform = transform * rasterio.Affine.translation(tile_x, tile_y)

            # Save the corresponding mask tile with the correct georeference
            output_tile_name = f"tile_{tile_x}_{tile_y}.tif"

            # Save the mask tile using rasterio
            profile = mask_img.profile
            profile.update({
                "width": tile_size,
                "height": tile_size,
                "transform": tile_transform,
                "crs": crs,
                "dtype": dtype
            })

            with rasterio.open(os.path.join(output_mask_path, output_tile_name), 'w', **profile) as dst:
                dst.write(mask_tile, 1)

            tile_count += 1

    print(f"Mask tiles saved in {output_mask_path}. Total tiles: {tile_count}")


# Paths
tiles_dir = r'C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\result\delft4'  # Path to the directory with tiles (e.g., 'tile_0_0.tif')
mask_path = r'D:\Super_Resolution\shp\buildings_delft_2_6.tif'  # Path to the original building mask
output_mask_path = r'C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\test_data\delft4_test_mask_tiles'  # Directory where the mask tiles will be saved

# Crop and save the mask tiles
crop_mask_based_on_tiles(tiles_dir, mask_path, output_mask_path)
