import os
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
from PIL import Image

hr_input = r"D:\Super_Resolution\Delft\HR\hr_1km_1km\Delft_hr_1km_1km_2_6.tif"
hr_output_dir = r"D:\Super_Resolution\Delft\HR\upscaled_tiles_512x512"
os.makedirs(hr_output_dir, exist_ok=True)

tile_size = 400
overlap_ratio = 0.10
overlap = int(tile_size * overlap_ratio)
scale_factor = 0.08 / 0.0625  # Upscaling from 8 cm to 6.25 cm (1.28x)

with rasterio.open(hr_input) as src:
    width, height = src.width, src.height
    stride = tile_size - overlap
    tile_count = 0

    for y in tqdm(range(0, height, stride)):
        for x in range(0, width, stride):
            if x + tile_size > width or y + tile_size > height:
                continue

            window = Window(x, y, tile_size, tile_size)
            tile = src.read(window=window).transpose(1, 2, 0)

            tile_pil = Image.fromarray(tile.astype(np.uint8))

            # Upscale from 400x400 (8 cm) to 512x512 (6.25 cm)
            upscaled_tile_pil = tile_pil.resize((512, 512), Image.BICUBIC)
            upscaled_tile = np.array(upscaled_tile_pil)

            base_transform = rasterio.windows.transform(window, src.transform)

            # Crop 256x256 patches from the upscaled tile
            for i in range(0, 512, 256):
                for j in range(0, 512, 256):
                    patch = upscaled_tile[i:i + 256, j:j + 256]
                    patch_filename = os.path.join(hr_output_dir, f"HR_tile_{x}_{y}_{i}_{j}.tif")

                    patch_transform = rasterio.Affine(
                        base_transform.a / scale_factor, base_transform.b, base_transform.c + (j * (base_transform.a / scale_factor)),
                        base_transform.d, base_transform.e / scale_factor, base_transform.f + (i * (base_transform.e / scale_factor))
                    )

                    profile = src.profile.copy()
                    profile.update({
                        "width": 256,
                        "height": 256,
                        "transform": patch_transform,
                        "dtype": "uint8",
                        "count": tile.shape[2]
                    })

                    with rasterio.open(patch_filename, "w", **profile) as dst:
                        dst.write(patch.transpose(2, 0, 1))

            tile_count += 1

print(f"HR tiling completed. {tile_count} tiles saved.")
