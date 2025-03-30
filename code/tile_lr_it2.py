import os
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from tqdm import tqdm


def tile_real_lr_iteration2(city, input_path, output_dir):
    """
    Tile the real 25cm LR raster into 64x64 tiles with 25% overlap.
    """
    os.makedirs(output_dir, exist_ok=True)

    tile_size = 64
    overlap = int(tile_size * 0.25)
    stride = tile_size - overlap
    tile_count = 0

    with rasterio.open(input_path) as src:
        width, height = src.width, src.height
        transform = src.transform

        for y in tqdm(range(0, height - tile_size + 1, stride), desc=f"Tiling LR {city}"):
            for x in range(0, width - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                tile = src.read(window=window)
                tile_transform = window_transform(window, transform)

                tile_path = os.path.join(output_dir, f"LR_tile_{tile_count}.tif")
                profile = src.profile.copy()
                profile.update({
                    "width": tile_size,
                    "height": tile_size,
                    "transform": tile_transform
                })

                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(tile)

                tile_count += 1

    print(f"Tiled {tile_count} LR tiles for {city}.")
