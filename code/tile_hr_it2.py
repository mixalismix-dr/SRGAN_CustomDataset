import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from PIL import Image
from tqdm import tqdm


def tile_upsampled_hr_iteration2(city, input_path, output_dir):
    """
    Upsample the HR raster to 6 cm and tile into 256x256 with 25% overlap.
    """
    os.makedirs(output_dir, exist_ok=True)

    tile_size = 256
    overlap = int(tile_size * 0.25)
    stride = tile_size - overlap

    # Temporary upsampled raster path
    upsampled_path = os.path.join(output_dir, f"{city}_hr_6cm_temp.tif")

    # Step 1: Upsample HR raster
    with rasterio.open(input_path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs
        profile = src.profile

        image_np = data[:3].transpose(1, 2, 0).astype("uint8")
        image_pil = Image.fromarray(image_np)
        new_size = (int(image_pil.width * 1.3333), int(image_pil.height * 1.3333))
        upsampled_pil = image_pil.resize(new_size, Image.BICUBIC)
        upsampled_np = np.array(upsampled_pil).transpose(2, 0, 1)

        profile.update({
            "width": new_size[0],
            "height": new_size[1],
            "transform": transform * transform.scale(8 / 6),
            "dtype": "uint8"
        })

        with rasterio.open(upsampled_path, "w", **profile) as dst:
            dst.write(upsampled_np)

    # Step 2: Tile the upsampled HR raster
    tile_count = 0
    with rasterio.open(upsampled_path) as src:
        width, height = src.width, src.height
        transform = src.transform

        for y in tqdm(range(0, height - tile_size + 1, stride), desc=f"Tiling HR {city}"):
            for x in range(0, width - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                tile = src.read(window=window)
                tile_transform = window_transform(window, transform)

                tile_path = os.path.join(output_dir, f"HR_tile_{tile_count}.tif")
                profile = src.profile.copy()
                profile.update({
                    "width": tile_size,
                    "height": tile_size,
                    "transform": tile_transform
                })

                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(tile)

                tile_count += 1

    os.remove(upsampled_path)
    print(f"Tiled {tile_count} upsampled HR tiles for {city}.")
