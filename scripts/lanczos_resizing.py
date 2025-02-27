import os
import argparse
import rasterio
import numpy as np
from PIL import Image


def resize_images(input_dir, output_dir, scale=4, resampling="lanczos", mode="down"):
    """Resize georeferenced RGB TIFF images while preserving CRS using PIL Lanczos resampling."""

    os.makedirs(output_dir, exist_ok=True)

    # Check if the input directory cotains subdirectories or just images
    contains_subdirs = any(os.path.isdir(os.path.join(input_dir, d)) for d in os.listdir(input_dir))

    if contains_subdirs:
        # Iterate through subdirectories
        for category in os.listdir(input_dir):
            category_path = os.path.join(input_dir, category)
            if not os.path.isdir(category_path):
                continue  # Skip if not a directory

            output_category_path = os.path.join(output_dir, category)
            os.makedirs(output_category_path, exist_ok=True)
            process_images(category_path, output_category_path, scale, resampling, mode)

    else:
        # If no subdirectories, process images directly in the input directory
        process_images(input_dir, output_dir, scale, resampling, mode)


def process_images(input_dir, output_dir, scale, resampling, mode):
    """Processes images inside a given directory and saves resized versions."""

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".tif"):
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.replace(".tif", "_down.tif") if mode == "down" else filename.replace(".tif",
                                                                                                            "_up.tif")
            output_path = os.path.join(output_dir, output_filename)

            with rasterio.open(input_path) as src:
                transform = src.transform
                crs = src.crs
                profile = src.profile
                image_data = src.read()  # Shape: (bands, height, width)

                # Convert to PIL Image (RGB)
                image_data = np.moveaxis(image_data, 0, -1)  # Convert (bands, H, W) -> (H, W, C)
                pil_image = Image.fromarray(image_data.astype(np.uint8))  # Ensure 8-bit RGB

                # Compute new dimensions
                if mode == "down":
                    new_size = (pil_image.width // scale, pil_image.height // scale)
                    new_transform = transform * transform.scale(scale)
                elif mode == "up":
                    new_size = (pil_image.width * scale, pil_image.height * scale)
                    new_transform = transform * transform.scale(1 / scale)
                else:
                    raise ValueError("Invalid mode. Choose 'down' or 'up'.")

                # Resize with PIL using Lanczos
                resized_pil = pil_image.resize(new_size, Image.LANCZOS)

                # Convert back to NumPy array
                resized_data = np.array(resized_pil).astype(np.uint8)  # Convert back to NumPy

                # Move back to Rasterio format (bands, H, W)
                resized_data = np.moveaxis(resized_data, -1, 0)

                # Update profile
                profile.update({
                    "height": new_size[1],
                    "width": new_size[0],
                    "transform": new_transform,
                    "crs": crs,
                    "dtype": "uint8",
                    "count": 3  # RGB bands
                })

                # Save the resized image
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(resized_data)

            print(f"Saved: {output_path} (CRS preserved)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize georeferenced RGB images while keeping CRS.")

    parser.add_argument("--input_dir", type=str, default=r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\test_data\delft4",
                        help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, default=r"D:\Super_Resolution\Delft\HR\generated_hr_normal_upscale\tiles_256",
                        help="Path to the output directory for resized images.")
    parser.add_argument("--scale", type=int, default=4, help="Scaling factor (default=4).")
    parser.add_argument("--mode", type=str, choices=["down", "up"], default="up",
                        help="Mode: 'down' for downsampling, 'up' for upsampling (default='down').")

    args = parser.parse_args()
    resize_images(args.input_dir, args.output_dir, args.scale, "lanczos", args.mode)
