import os
import rasterio
import numpy as np
from PIL import Image


def resize_images(input_dir, output_dir, scale=4, resampling="bicubic", mode="down"):
    """
    Downsamples or upsamples georeferenced RGB TIFFs in input_dir, preserving CRS and filenames.
    Converts HR_ prefix to LR_ in output names.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".tif"):
            continue

        input_path = os.path.join(input_dir, filename)

        # Output filename: HR_tile_00001_delft.tif -> LR_tile_00001_delft.tif
        output_filename = filename.replace("HR_", "LR_")
        output_path = os.path.join(output_dir, output_filename)

        with rasterio.open(input_path) as src:
            transform = src.transform
            crs = src.crs
            profile = src.profile
            image_data = src.read()  # (bands, H, W)

            # Convert to PIL RGB image
            image_data = np.moveaxis(image_data, 0, -1)
            pil_image = Image.fromarray(image_data.astype(np.uint8))

            if mode == "down":
                new_size = (pil_image.width // scale, pil_image.height // scale)
                new_transform = transform * transform.scale(scale)
            elif mode == "up":
                new_size = (pil_image.width * scale, pil_image.height * scale)
                new_transform = transform * transform.scale(1 / scale)
            else:
                raise ValueError("Mode must be 'down' or 'up'")

            resized_pil = pil_image.resize(new_size, Image.BICUBIC)
            resized_data = np.array(resized_pil).astype(np.uint8)
            resized_data = np.moveaxis(resized_data, -1, 0)

            profile.update({
                "height": new_size[1],
                "width": new_size[0],
                "transform": new_transform,
                "crs": crs,
                "dtype": "uint8",
                "count": 3
            })

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(resized_data)

        print(f"Saved: {output_path}")
