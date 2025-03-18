from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

# Load images
image1 = cv2.imread('HR_tile_569.tif')  # Ground truth
image2 = cv2.imread('resampled_569.tif')  # Distorted or SR image

# Convert images to grayscale or ensure they have the same format
if image1.shape != image2.shape:
    raise ValueError("Images must have the same dimensions to calculate PSNR")

# Compute PSNR
psnr_value = psnr(image1, image2, data_range=255)
print(f"PSNR: {psnr_value:.2f} dB")


