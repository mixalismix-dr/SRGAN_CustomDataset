from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

# Load images
image1 = cv2.imread('ground_truth_hr_0_0.tif')        # Ground truth
image2 = cv2.imread('resampled_0_0.tif')              # Bicubic
image3 = cv2.imread('super_res_8cm.tif')              # SR result

# Get the minimum common height and width
min_h = min(image1.shape[0], image2.shape[0], image3.shape[0])
min_w = min(image1.shape[1], image2.shape[1], image3.shape[1])

# Crop all images to the common shape (top-left crop)
image1 = image1[:min_h, :min_w]
image2 = image2[:min_h, :min_w]
image3 = image3[:min_h, :min_w]

# Confirm shape match
assert image1.shape == image2.shape == image3.shape, "Images must have the same dimensions"

# PSNR
psnr_bicubic = psnr(image1, image2, data_range=255)
psnr_sr = psnr(image1, image3, data_range=255)

# SSIM (handle version compatibility)
try:
    ssim_bicubic = ssim(image1, image2, data_range=255, channel_axis=-1)  # skimage >= 0.19
    ssim_sr = ssim(image1, image3, data_range=255, channel_axis=-1)
except TypeError:
    ssim_bicubic = ssim(image1, image2, data_range=255, multichannel=True)  # skimage < 0.19
    ssim_sr = ssim(image1, image3, data_range=255, multichannel=True)

# Print results
print(f"Bicubic PSNR: {psnr_bicubic:.2f} dB")
print(f"Bicubic SSIM: {ssim_bicubic:.4f}")
print(f"SR PSNR: {psnr_sr:.2f} dB")
print(f"SR SSIM: {ssim_sr:.4f}")
print(image1.shape, image2.shape, image3.shape)