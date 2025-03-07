import os
import glob
import numpy as np
import cv2
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from tqdm import tqdm
from tabulate import tabulate
import torch
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"is cuda available: {torch.cuda.is_available()}")
lpips_fn = lpips.LPIPS(net='vgg').to(device)

def compute_lpips(hr_array, sr_array):
    hr_tensor = torch.tensor(hr_array / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    sr_tensor = torch.tensor(sr_array / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    hr_tensor = hr_tensor * 2 - 1
    sr_tensor = sr_tensor * 2 - 1
    return lpips_fn(sr_tensor, hr_tensor).item()

def compute_metrics(hr_array, sr_array, up_array):
    """Compute PSNR, SSIM, and BRISQUE metrics for SR and UP images."""
    # Convert to grayscale (Y-channel in YCbCr)
    y_hr = rgb2ycbcr(hr_array)[:, :, 0]
    y_sr = rgb2ycbcr(sr_array)[:, :, 0]
    y_up = rgb2ycbcr(up_array)[:, :, 0]


    # Compute PSNR
    psnr_sr = peak_signal_noise_ratio(y_hr, y_sr, data_range=255.0)
    psnr_up = peak_signal_noise_ratio(y_hr, y_up, data_range=255.0)

    # Compute SSIM
    edges_sr = cv2.Canny(y_sr.astype(np.uint8), 100, 200)
    edges_up = cv2.Canny(y_up.astype(np.uint8), 100, 200)
    edges_hr = cv2.Canny(y_hr.astype(np.uint8), 100, 200)
    ssim_sr = structural_similarity(edges_sr, edges_hr, data_range=255.0)
    ssim_up = structural_similarity(edges_hr, edges_up, data_range=255.0)


    return psnr_sr, psnr_up, ssim_sr, ssim_up


def process_category(category_dir, sr_path, up_root_path):
    """Process all images in a category and compute metrics."""
    hr_images = sorted(glob.glob(os.path.join(category_dir, "*.tif")))

    lpips_sr_values, lpips_up_values = [], []
    psnr_sr_values, psnr_up_values = [], []
    ssim_sr_values, ssim_up_values = [], []


    for hr_img in tqdm(hr_images, total=len(hr_images), desc=f"Processing {os.path.basename(category_dir)}"):
        tile_name = os.path.basename(hr_img).replace(".tif", "")

        # Find matching SR and UP images
        sr_candidates = glob.glob(os.path.join(sr_path, f"res_{tile_name}_down.tif"))
        up_candidates = glob.glob(os.path.join(up_root_path, "**", f"{tile_name}_down_up.tif"), recursive=True)

        if not sr_candidates or not up_candidates:
            continue

        sr_img = sr_candidates[0]
        up_img = up_candidates[0]

        # Read images
        hr_array = cv2.imread(hr_img, cv2.IMREAD_COLOR)
        sr_array = cv2.imread(sr_img, cv2.IMREAD_COLOR)
        up_array = cv2.imread(up_img, cv2.IMREAD_COLOR)

        if hr_array is None or sr_array is None or up_array is None:
            print(f"Skipping {tile_name}: Unable to read one or more images")
            continue

        # Compute metrics
        psnr_sr, psnr_up, ssim_sr, ssim_up= compute_metrics(hr_array, sr_array, up_array)


        #Compute LPIPS
        lpips_sr = compute_lpips(hr_array, sr_array)
        lpips_up = compute_lpips(hr_array, up_array)

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
            "avg_lpips_up": np.mean(lpips_up_values)

        }
    return None


def save_metrics_to_csv(category_metrics, output_metrics_file):
    """Save category-wise metrics to a CSV file."""
    with open(output_metrics_file, "w") as f:
        f.write("Category,Avg_PSNR_SR,Avg_PSNR_UP,Avg_SSIM_SR,Avg_SSIM_UP,LPIPS_SR, LPIPS_UP\n")
        for category, metrics in category_metrics.items():
            f.write(f"{category},{metrics['avg_psnr_sr']:.4f},{metrics['avg_psnr_up']:.4f},{metrics['avg_ssim_sr']:.4f},{metrics['avg_ssim_up']:.4f},"
                    f"{metrics['avg_lpips_sr']:.4f}, {metrics['avg_lpips_up']:.4f}\n")


def main():
    # Paths
    sr_path = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\result\delft3"
    hr_path = r"D:\Super_Resolution\Delft\HR\real_hr\tiles_256"
    up_root_path = r"D:\Super_Resolution\Delft\HR\generated_hr_normal_upscale\tiles_256_bic"
    output_metrics_file = "metrics_results_new.csv"

    # Initialize storage for category-based results
    category_metrics = {}

    # Iterate through subdirectories (categories)
    for category in os.listdir(hr_path):
        category_dir = os.path.join(hr_path, category)
        if not os.path.isdir(category_dir):
            continue  # Skip if not a directory

        # Process the category
        metrics = process_category(category_dir, sr_path, up_root_path)
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