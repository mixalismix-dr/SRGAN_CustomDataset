import os
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from tqdm import tqdm


def tile_raster(input_path, output_dir, city, tile_size=256, overlap_ratio=0.10):
    """
    Tiles a georeferenced raster into 256x256 tiles with overlap.
    Output is saved in output_dir as HR_tile_00001_<city>.tif
    """
    os.makedirs(output_dir, exist_ok=True)

    overlap = int(tile_size * overlap_ratio)
    stride = tile_size - overlap
    city = city.lower()
    tile_count = 1

    with rasterio.open(input_path) as src:
        width, height = src.width, src.height

        for y in tqdm(range(0, height - tile_size + 1, stride), desc=f"Tiling rows ({city})"):
            for x in range(0, width - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                transform = window_transform(window, src.transform)
                profile = src.profile.copy()
                profile.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": transform
                })

                tile_data = src.read(window=window)

                filename = f"HR_tile_{tile_count:05d}_{city}.tif"
                tile_path = os.path.join(output_dir, filename)

                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(tile_data)

                tile_count += 1

    print(f"\n Tiled {tile_count - 1} patches saved to {output_dir}")
