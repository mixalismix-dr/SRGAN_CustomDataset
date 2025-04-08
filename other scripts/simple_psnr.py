import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.exposure import match_histograms
import matplotlib
matplotlib.use("TkAgg")  # Or "QtAgg" if you have PyQt5/PySide installed
import matplotlib.pyplot as plt

# --- Input images ---
img1_path = r"D:\Super_Resolution\Hague\Iteration_2\HR_tiles_5_8_256x256\HR_tile_277.tif"  # Ground Truth (e.g., HR)
# img2_path = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\result\hague\res_LR_tile_277.tif"
img2_path = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\test_data\hague\LR_tile_resampled_277.tif" #Comparison (e.g., SR or Bicubic)

 # --- Load as RGB ---
img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

if img1 is None or img2 is None:
    raise FileNotFoundError("One of the images could not be read.")

# --- Convert BGR to RGB for plotting ---
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# --- Match size to smallest common shape ---
min_h = min(img1.shape[0], img2.shape[0])
min_w = min(img1.shape[1], img2.shape[1])

img1_cropped = img1[:min_h, :min_w]
img2_cropped = img2[:min_h, :min_w]

# --- Histogram match SR/Bicubic image to HR ---
# img2_matched = match_histograms(img2_cropped, img1_cropped, channel_axis=-1)
# img2_matched = np.clip(img2_matched, 0, 255).astype(np.uint8)

# --- Compute PSNR ---
score = psnr(img1_cropped, img2_cropped, data_range=255)
print(f"PSNR (after histogram matching): {score:.2f} dB")

# --- Plot the two images side by side ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1_cropped)
plt.title("HR Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img2_cropped)
plt.title("Matched SR/Bicubic")
plt.axis("off")

plt.tight_layout()
plt.show()