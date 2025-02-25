import os
import argparse
from PIL import Image

# Define available resampling methods
RESAMPLING_METHODS = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
}


def resize_images(input_root, output_root, scale=4, resampling="lanczos", mode="down"):
    """ Resizes images inside subdirectories while preserving the folder structure. """

    if resampling not in RESAMPLING_METHODS:
        raise ValueError(f"Invalid resampling method. Choose from: {list(RESAMPLING_METHODS.keys())}")

    for subdir in os.listdir(input_root):
        input_dir = os.path.join(input_root, subdir)
        output_dir = os.path.join(output_root, subdir)

        if not os.path.isdir(input_dir):
            continue  # Skip non-directory files

        os.makedirs(output_dir, exist_ok=True)  # Create corresponding subdirectory

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(".tif"):
                input_path = os.path.join(input_dir, filename)
                output_filename = filename.replace(".tif", f"_{resampling}_{mode}.tif")
                output_path = os.path.join(output_dir, output_filename)

                # Open image
                image = Image.open(input_path)

                # Compute new dimensions
                if mode == "down":
                    new_size = (image.width // scale, image.height // scale)
                elif mode == "up":
                    new_size = (image.width * scale, image.height * scale)
                else:
                    raise ValueError("Invalid mode. Choose 'down' or 'up'.")

                # Resize using selected resampling method
                resized_image = image.resize(new_size, RESAMPLING_METHODS[resampling])

                # Save the output image
                resized_image.save(output_path)

                print(f"Resized image saved at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images while maintaining directory structure.")

    parser.add_argument("--input_root", type=str, default=r"D:\Super_Resolution\Delft\HR\synthetic_lr_from_hr\tiles_64",
                        help="Root directory containing image subdirectories.")
    parser.add_argument("--output_root", type=str,
                        default=r"D:\Super_Resolution\Delft\HR\generated_hr_normal_upscale\tiles_256",
                        help="Root directory where resized images will be saved.")
    parser.add_argument("--scale", type=int, default=4, help="Scaling factor (default=4).")
    parser.add_argument("--resampling", type=str, choices=RESAMPLING_METHODS.keys(), default="lanczos",
                        help="Resampling method (default='lanczos').")
    parser.add_argument("--mode", type=str, choices=["down", "up"], default="up",
                        help="Mode: 'down' for downsampling, 'up' for upsampling (default='down').")

    args = parser.parse_args()
    resize_images(args.input_root, args.output_root, args.scale, args.resampling, args.mode)
