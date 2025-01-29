import os
import cv2
import numpy as np
from utils_blindsr import degradation_bsrgan
from utils_image import imread_uint, uint2single, imsave

# Paths
input_dir = "input_images"  # Change to your folder with high-res images
output_dir = "output_LR_images"
os.makedirs(output_dir, exist_ok=True)

scale_factor = 4  # Set your downscaling factor (e.g., 2, 3, 4)

# Process each image
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)

    # Read image
    img = imread_uint(img_path, n_channels=3)  # Read as RGB
    img = uint2single(img)  # Normalize to [0, 1]

    # Apply BSR degradation
    img_LR, img_HR = degradation_bsrgan(img, sf=scale_factor, lq_patchsize=72)

    # Save the degraded image
    output_path = os.path.join(output_dir, f"LR_{img_name}")
    imsave(img_LR * 255, output_path)

    print(f"Saved: {output_path}")

print("Downsampling complete!")
