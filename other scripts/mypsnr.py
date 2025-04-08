import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from tabulate import tabulate

# === INPUT PATHS ===
hr_dir = r"D:\Super_Resolution\Zwolle\Iteration_2\HR_tiles_4_3_256x256"
bicubic_dir = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\test_data\p3\LR_tile_resampled_0_0.tif"
sr_dir = r"D:\Super_Resolution\SRGAN_CustomDataset\result\p3"

# === STORAGE ===
psnr_sr_all = []
psnr_bicubic_all = []
ssim_sr_all = []
ssim_bicubic_all = []

# === LOOP ===
hr_images = sorted([f for f in os.listdir(hr_dir) if f.endswith(".tif")])

for filename in tqdm(hr_images):
    base = filename.replace(".tif", "")
    hr_path = os.path.join(hr_dir, filename)
    bicubic_path = os.path.join(bicubic_dir, base + "_down.tif")
    sr_path = os.path.join(sr_dir, "res_" + base + "_down.tif")

    if not os.path.exists(bicubic_path) or not os.path.exists(sr_path):
        print(f"Skipping {base}: missing bicubic or SR image")
        continue

    hr = cv2.imread(hr_path)
    bicubic = cv2.imread(bicubic_path)
    sr = cv2.imread(sr_path)

    if hr is None or bicubic is None or sr is None:
        print(f"Skipping {base}: unreadable image")
        continue

    # Crop to smallest common size
    min_h = min(hr.shape[0], bicubic.shape[0], sr.shape[0])
    min_w = min(hr.shape[1], bicubic.shape[1], sr.shape[1])
    hr = hr[:min_h, :min_w]
    bicubic = bicubic[:min_h, :min_w]
    sr = sr[:min_h, :min_w]

    # Compute metrics
    psnr_sr = psnr(hr, sr, data_range=255)
    psnr_bicubic = psnr(hr, bicubic, data_range=255)

    try:
        ssim_sr = ssim(hr, sr, data_range=255, channel_axis=-1)
        ssim_bicubic = ssim(hr, bicubic, data_range=255, channel_axis=-1)
    except TypeError:
        ssim_sr = ssim(hr, sr, data_range=255, multichannel=True)
        ssim_bicubic = ssim(hr, bicubic, data_range=255, multichannel=True)

    # Store
    psnr_sr_all.append(psnr_sr)
    psnr_bicubic_all.append(psnr_bicubic)
    ssim_sr_all.append(ssim_sr)
    ssim_bicubic_all.append(ssim_bicubic)

# === AVERAGE RESULTS ===
table = [
    ["PSNR (SR)", f"{np.mean(psnr_sr_all):.2f} dB"],
    ["PSNR (Bicubic)", f"{np.mean(psnr_bicubic_all):.2f} dB"],
    ["SSIM (SR)", f"{np.mean(ssim_sr_all):.4f}"],
    ["SSIM (Bicubic)", f"{np.mean(ssim_bicubic_all):.4f}"]
]

print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
