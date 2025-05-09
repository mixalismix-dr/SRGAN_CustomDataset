import os
import glob
import numpy as np
import cv2
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from tabulate import tabulate
import torch
import lpips
from PIL import Image
import rasterio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='vgg').to(device)

def compute_lpips(hr_array, sr_array):
    """Compute LPIPS (Perceptual Loss) with shape alignment."""
    # Resize to common dimensions (min height & width)
    min_h = min(hr_array.shape[0], sr_array.shape[0])
    min_w = min(hr_array.shape[1], sr_array.shape[1])

    hr_array = hr_array[:min_h, :min_w]
    sr_array = sr_array[:min_h, :min_w]

    hr_tensor = torch.tensor(hr_array / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    sr_tensor = torch.tensor(sr_array / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    hr_tensor = hr_tensor * 2 - 1
    sr_tensor = sr_tensor * 2 - 1
    return lpips_fn(sr_tensor, hr_tensor).item()


def compute_metrics(hr_array, sr_array, up_array):
    """Compute PSNR, SSIM for SR and upsampled images."""
    min_h = min(hr_array.shape[0], sr_array.shape[0], up_array.shape[0])
    min_w = min(hr_array.shape[1], sr_array.shape[1], up_array.shape[1])

    hr_array = hr_array[:min_h, :min_w]
    sr_array = sr_array[:min_h, :min_w]
    up_array = up_array[:min_h, :min_w]

    y_hr = rgb2ycbcr(hr_array)[:, :, 0]
    y_sr = rgb2ycbcr(sr_array)[:, :, 0]
    y_up = rgb2ycbcr(up_array)[:, :, 0]

    # Compute PSNR
    psnr_sr = peak_signal_noise_ratio(y_hr, y_sr, data_range=255.0)
    psnr_up = peak_signal_noise_ratio(y_hr, y_up, data_range=255.0)

    # Compute SSIM on edge maps
    edges_sr = cv2.Canny(y_sr.astype(np.uint8), 100, 200)
    edges_up = cv2.Canny(y_up.astype(np.uint8), 100, 200)
    edges_hr = cv2.Canny(y_hr.astype(np.uint8), 100, 200)
    ssim_sr = structural_similarity(edges_sr, edges_hr, data_range=255.0)
    ssim_up = structural_similarity(edges_hr, edges_up, data_range=255.0)

    return psnr_sr, psnr_up, ssim_sr, ssim_up

def upsample_bicubic(lr_image):
    """Upsample LR image dynamically using bicubic interpolation to match HR resolution."""
    lr_pil = Image.fromarray(lr_image.astype(np.uint8))
    new_size = (lr_pil.width * 4, lr_pil.height * 4)  # Scale x4
    upscaled_pil = lr_pil.resize(new_size, Image.BICUBIC)
    return np.array(upscaled_pil)

def process_category(category_dir, sr_path, lr_path):
    """Process all images in a category and compute metrics."""
    hr_images = sorted(glob.glob(os.path.join(category_dir, "*.tif")))

    lpips_sr_values, lpips_up_values = [], []
    psnr_sr_values, psnr_up_values = [], []
    ssim_sr_values, ssim_up_values = [], []

    for hr_img in tqdm(hr_images, total=len(hr_images), desc=f"Processing {os.path.basename(category_dir)}"):
        tile_name = os.path.basename(hr_img).replace(".tif", "")  # e.g., HR_tile_0
        tile_number = tile_name.replace("HR_tile_", "")  # just the number part
        sr_filename = f"res_LR_tile_{tile_number}.tif"
        lr_filename = f"LR_tile_{tile_number}.tif"

        sr_candidates = glob.glob(os.path.join(sr_path, sr_filename))
        lr_candidates = glob.glob(os.path.join(lr_path, lr_filename))

        if not sr_candidates or not lr_candidates:
            continue

        sr_img = sr_candidates[0]
        lr_img = lr_candidates[0]

        # Read images
        hr_array = cv2.imread(hr_img, cv2.IMREAD_COLOR)
        sr_array = cv2.imread(sr_img, cv2.IMREAD_COLOR)
        lr_array = cv2.imread(lr_img, cv2.IMREAD_COLOR)

        if hr_array is None or sr_array is None or lr_array is None:
            print(f"Skipping {tile_name}: Unable to read one or more images")
            continue

        # Upsample LR image dynamically
        upscaled_lr_array = upsample_bicubic(lr_array)

        # Compute metrics
        psnr_sr, psnr_up, ssim_sr, ssim_up = compute_metrics(hr_array, sr_array, upscaled_lr_array)

        # Compute LPIPS
        lpips_sr = compute_lpips(hr_array, sr_array)
        lpips_up = compute_lpips(hr_array, upscaled_lr_array)

        # Store values
        lpips_sr_values.append(lpips_sr)
        lpips_up_values.append(lpips_up)
        psnr_sr_values.append(psnr_sr)
        psnr_up_values.append(psnr_up)
        ssim_sr_values.append(ssim_sr)
        ssim_up_values.append(ssim_up)

    # Compute category-wise averages
    if psnr_sr_values:
        return {
            "avg_psnr_sr": np.mean(psnr_sr_values),
            "avg_psnr_up": np.mean(psnr_up_values),
            "avg_ssim_sr": np.mean(ssim_sr_values),
            "avg_ssim_up": np.mean(ssim_up_values),
            "avg_lpips_sr": np.mean(lpips_sr_values),
            "avg_lpips_up": np.mean(lpips_up_values),
        }
    return None

def save_metrics_to_csv(category_metrics, output_metrics_file):
    """Save category-wise metrics to a CSV file."""
    with open(output_metrics_file, "w") as f:
        f.write("Category,Avg_PSNR_SR,Avg_PSNR_UP,Avg_SSIM_SR,Avg_SSIM_UP,LPIPS_SR,LPIPS_UP\n")
        for category, metrics in category_metrics.items():
            f.write(f"{category},{metrics['avg_psnr_sr']:.4f},{metrics['avg_psnr_up']:.4f},{metrics['avg_ssim_sr']:.4f},{metrics['avg_ssim_up']:.4f},"
                    f"{metrics['avg_lpips_sr']:.4f}, {metrics['avg_lpips_up']:.4f}\n")

def main():
    # Paths
    sr_path = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\result\hague"
    hr_path = r"D:\Super_Resolution\Hague\Iteration_2\HR_tiles_5_8_256x256"
    lr_path = r"D:\Super_Resolution\Hague\Iteration_2\LR_tiles_5_8_64x64"
    output_metrics_file = "metrics_results_new.csv"

    # Initialize storage for category-based results
    category_metrics = {}

    # Iterate through subdirectories (categories)
    for category in os.listdir(hr_path):
        category_dir = os.path.join(hr_path, category)
        if not os.path.isdir(category_dir):
            continue  # Skip if not a directory

        # Process the category
        metrics = process_category(category_dir, sr_path, lr_path)
        if metrics:
            category_metrics[category] = metrics

    # Save results to CSV
    save_metrics_to_csv(category_metrics, output_metrics_file)

    # Print results in a table format
    table_headers = [
        "Category", "PSNR(SR)", "PSNR(UP)", "SSIM(SR)", "SSIM(UP)",
        "LPIPS(SR)", "LPIPS(UP)"
    ]
    table_data = []

    for category, metrics in category_metrics.items():
        table_data.append([
            category,
            f"{metrics['avg_psnr_sr']:.4f}", f"{metrics['avg_psnr_up']:.4f}",
            f"{metrics['avg_ssim_sr']:.4f}", f"{metrics['avg_ssim_up']:.4f}",
            f"{metrics['avg_lpips_sr']:.4f}", f"{metrics['avg_lpips_up']:.4f}"
        ])

    print("\nCategory-wise results:")
    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))

    # Print footnotes for metric interpretation
    print("\n" + "-" * 80)
    print("Metric Interpretation:")
    print("• PSNR: Higher is better. If PSNR(SR) > PSNR(UP), SR performs better than simple upscaling.")
    print("• SSIM: Higher means more perceptual similarity to ground truth. If SSIM(SR) > SSIM(UP), SR is preferable.")
    print("• LPIPS(SR): Lower values mean higher perceptual similarity to the ground truth. If LPIPS(SR) < LPIPS(UP), "
          "the SR method gives more perceptually accurate results.")
    print("-" * 80)


if __name__ == "__main__":
    main()
