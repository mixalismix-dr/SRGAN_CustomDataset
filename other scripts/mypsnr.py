from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

# Load images
image1 = cv2.imread(r"C:\Users\mike_\OneDrive\Desktop\real_lr_forpsnr.tif")  # Ground truth
image2 = cv2.imread(r"/result/geores_0000.tif")  # Distorted or SR image

# Convert images to grayscale or ensure they have the same format
if image1.shape != image2.shape:
    raise ValueError("Images must have the same dimensions to calculate PSNR")

# Compute PSNR
psnr_value = psnr(image1, image2, data_range=255)
print(f"PSNR: {psnr_value:.2f} dB")


