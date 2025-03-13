import os
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
from PIL import Image

# Paths
hr_input = r"D:\Super_Resolution\Delft\HR\hr_1km_1km\Delft_hr_1km_1km_2_6.tif"
lr_input = r"D:\Super_Resolution\Delft\LR\lr_1km_1km\Delft_lr_1km_1km_2_6.tif"
hr_output_dir = r"D:\Super_Resolution\Delft\HR\upscaled_tiles_256x256"
lr_output_dir = r"D:\Super_Resolution\Delft\LR\real_lr\tiles_64x64"
os.makedirs(hr_output_dir, exist_ok=True)
os.makedirs(lr_output_dir, exist_ok=True)

# HR Parameters
hr_tile_size = 400  # HR tile size before upscaling
scale_factor = 0.08 / 0.0625  # 8cm -> 6.25cm (1.28x)

# LR Parameters
lr_tile_size = 64  # LR tile size
lr_scale_factor = 25 / 6.25  # 25cm -> 6.25cm

with rasterio.open(hr_input) as hr_src, rasterio.open(lr_input) as lr_src:
    width, height = hr_src.width, hr_src.height
    hr_transform = hr_src.transform
    lr_transform = lr_src.transform

    # Iterate through HR raster and extract matching LR tiles
    for y in tqdm(range(0, height, hr_tile_size)):
        for x in range(0, width, hr_tile_size):
            if x + hr_tile_size > width or y + hr_tile_size > height:
                continue

            # **Process HR Tile**
            hr_window = Window(x, y, hr_tile_size, hr_tile_size)
            hr_tile = hr_src.read(window=hr_window).transpose(1, 2, 0)

            hr_tile_pil = Image.fromarray(hr_tile.astype(np.uint8))

            # Upscale HR from 400x400 (8 cm) to 512x512 (6.25 cm)
            upscaled_tile_pil = hr_tile_pil.resize((512, 512), Image.BICUBIC)
            upscaled_tile = np.array(upscaled_tile_pil)

            base_transform = rasterio.windows.transform(hr_window, hr_transform)

            # Crop 256x256 patches from upscaled HR tile
            for i in range(0, 512, 256):
                for j in range(0, 512, 256):
                    patch = upscaled_tile[i:i + 256, j:j + 256]

                    # **Generate tile name using spatial coordinates**
                    tile_name = f"tile_{x}_{y}_{i}_{j}.tif"

                    hr_patch_filename = os.path.join(hr_output_dir, f"HR_{tile_name}")

                    patch_transform = rasterio.Affine(
                        base_transform.a / scale_factor, base_transform.b, base_transform.c + (j * (base_transform.a / scale_factor)),
                        base_transform.d, base_transform.e / scale_factor, base_transform.f + (i * (base_transform.e / scale_factor))
                    )

                    hr_profile = hr_src.profile.copy()
                    hr_profile.update({
                        "width": 256,
                        "height": 256,
                        "transform": patch_transform,
                        "dtype": "uint8",
                        "count": hr_tile.shape[2]
                    })

                    with rasterio.open(hr_patch_filename, "w", **hr_profile) as dst:
                        dst.write(patch.transpose(2, 0, 1))

            # **Process Corresponding LR Tile (Ensure Same Name)**
            # Convert HR tile (x, y) to world coordinates
            hr_x, hr_y = hr_transform * (x, y)

            # Convert to LR grid
            aligned_x, aligned_y = ~lr_transform * (hr_x, hr_y)
            aligned_x, aligned_y = int(aligned_x), int(aligned_y)

            lr_window = Window(aligned_x, aligned_y, lr_tile_size, lr_tile_size)
            lr_tile = lr_src.read(window=lr_window)

            lr_tile_filename = os.path.join(lr_output_dir, f"LR_{tile_name}")
            lr_profile = lr_src.profile.copy()
            lr_profile.update({
                "width": lr_tile_size,
                "height": lr_tile_size,
                "transform": rasterio.windows.transform(lr_window, lr_transform)
            })

            with rasterio.open(lr_tile_filename, "w", **lr_profile) as dst:
                dst.write(lr_tile)

print(f"Processing completed: {len(os.listdir(hr_output_dir))} HR tiles, {len(os.listdir(lr_output_dir))} LR tiles saved.")
