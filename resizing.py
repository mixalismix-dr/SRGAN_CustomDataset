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


def resize_images(input_dir, output_dir, scale=4, resampling="lanczos", mode="down"):
    """ Resizes images in a directory using a specified resampling method.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory for resized images.
        scale (int): Scaling factor (default=4).
        resampling (str): Resampling method (default="lanczos").
        mode (str): "down" for downsampling, "up" for upsampling.
    """

    if resampling not in RESAMPLING_METHODS:
        raise ValueError(f"Invalid resampling method. Choose from: {list(RESAMPLING_METHODS.keys())}")

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists

    # Process all TIFF images in the directory
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
    parser = argparse.ArgumentParser(description="Resize images using various resampling methods.")

    parser.add_argument("--input_dir", type=str, default=r"D:\Super_Resolution\Delft\HR\real_hr\tiles_256\test",
                        help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, default=r"D:\Super_Resolution\Delft\HR\real_hr\tiles_256\resized",
                        help="Path to the output directory for resized images.")
    parser.add_argument("--scale", type=int, default=4, help="Scaling factor (default=4).")
    parser.add_argument("--resampling", type=str, choices=RESAMPLING_METHODS.keys(), default="lanczos",
                        help="Resampling method (default='lanczos').")
    parser.add_argument("--mode", type=str, choices=["down", "up"], default="down",
                        help="Mode: 'down' for downsampling, 'up' for upsampling (default='down').")

    args = parser.parse_args()

    resize_images(args.input_dir, args.output_dir, args.scale, args.resampling, args.mode)
