import os
import glob
import numpy as np
import cv2
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from tqdm import tqdm
from tabulate import tabulate

def compute_blur(image_path):
    """Compute blur score using variance of Laplacian."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return cv2.Laplacian(img, cv2.CV_64F).var()

def compute_AG(image_path):
    """Compute Average Gradient (AG) of an image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Compute gradients using Sobel operator
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Compute average gradient (AG)
    ag = np.mean(gradient_magnitude)
    return ag


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
    ssim_sr = structural_similarity(y_hr, y_sr, data_range=255.0)
    ssim_up = structural_similarity(y_hr, y_up, data_range=255.0)

    # Compute BRISQUE
    brisque_sr = cv2.quality.QualityBRISQUE_compute(sr_array, "brisque_model_live.yml", "brisque_range_live.yml")
    brisque_up = cv2.quality.QualityBRISQUE_compute(up_array, "brisque_model_live.yml", "brisque_range_live.yml")

    # Compute RMSE
    rmse_sr = np.sqrt(mean_squared_error(y_hr, y_sr))
    rmse_up = np.sqrt(mean_squared_error(y_hr, y_up))

    return psnr_sr, psnr_up, ssim_sr, ssim_up, brisque_sr, brisque_up, rmse_up, rmse_sr


def process_category(category_dir, sr_path, up_root_path):
    """Process all images in a category and compute metrics."""
    hr_images = sorted(glob.glob(os.path.join(category_dir, "*.tif")))

    psnr_sr_values, psnr_up_values = [], []
    ssim_sr_values, ssim_up_values = [], []
    brisque_sr_values, brisque_up_values = [], []
    ag_sr_values, ag_up_values, ag_hr_values = [], [], []
    blur_sr_values, blur_up_values, blur_hr_values = [], [], []
    rmse_sr_values, rmse_up_values = [], []

    for hr_img in tqdm(hr_images, total=len(hr_images), desc=f"Processing {os.path.basename(category_dir)}"):
        tile_name = os.path.basename(hr_img).replace(".tif", "")

        # Find matching SR and UP images
        sr_candidates = glob.glob(os.path.join(sr_path, f"res_{tile_name}_down.tif"))
        up_candidates = glob.glob(os.path.join(up_root_path, "**", f"{tile_name}_down_up.tif"), recursive=True)

        if not sr_candidates or not up_candidates:
            print(f"Warning: No SR/UP image found for {tile_name}, skipping...")
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
        psnr_sr, psnr_up, ssim_sr, ssim_up, brisque_sr, brisque_up, rmse_up, rmse_sr = compute_metrics(hr_array, sr_array, up_array)

        # Compute Average Gradient (AG)
        ag_hr = compute_AG(hr_img)
        ag_sr = compute_AG(sr_img)
        ag_up = compute_AG(up_img)

        # Compute Blur Score
        blur_sr = compute_blur(sr_img)
        blur_up = compute_blur(up_img)

        # Store values
        psnr_sr_values.append(psnr_sr)
        psnr_up_values.append(psnr_up)
        ssim_sr_values.append(ssim_sr)
        ssim_up_values.append(ssim_up)
        brisque_sr_values.append(brisque_sr)
        brisque_up_values.append(brisque_up)
        ag_hr_values.append(ag_hr)
        ag_sr_values.append(ag_sr)
        ag_up_values.append(ag_up)
        rmse_up_values.append(rmse_up)
        rmse_sr_values.append(rmse_sr)
        blur_sr_values.append(blur_sr)
        blur_up_values.append(blur_up)

    # Compute category-wise averages
    if psnr_sr_values:
        return {
            "avg_psnr_sr": np.mean(psnr_sr_values),
            "avg_psnr_up": np.mean(psnr_up_values),
            "avg_ssim_sr": np.mean(ssim_sr_values),
            "avg_ssim_up": np.mean(ssim_up_values),
            "avg_brisque_sr": np.mean(brisque_sr_values),
            "avg_brisque_up": np.mean(brisque_up_values),
            "avg_ag_hr": np.mean(ag_hr_values),
            "avg_ag_sr": np.mean(ag_sr_values),
            "avg_ag_up": np.mean(ag_up_values),
            "avg_rmse_sr": np.mean(rmse_sr_values),
            "avg_rmse_up": np.mean(rmse_up_values),
            "avg_blur_sr": np.mean(blur_sr_values),
            "avg_blur_up": np.mean(blur_up_values),

        }
    return None


def save_metrics_to_csv(category_metrics, output_metrics_file):
    """Save category-wise metrics to a CSV file."""
    with open(output_metrics_file, "w") as f:
        f.write("Category,Avg_PSNR_SR,Avg_PSNR_UP,Avg_SSIM_SR,Avg_SSIM_UP,Avg_BRISQUE_SR,Avg_BRISQUE_UP,Avg_AG_HR,Avg_AG_SR,Avg_AG_UP, RMSE_SR, RMSE_UP, Blur_SR, Blur_UP\n")
        for category, metrics in category_metrics.items():
            f.write(f"{category},{metrics['avg_psnr_sr']:.4f},{metrics['avg_psnr_up']:.4f},{metrics['avg_ssim_sr']:.4f},{metrics['avg_ssim_up']:.4f},"
                    f"{metrics['avg_brisque_sr']:.4f},{metrics['avg_brisque_up']:.4f},{metrics['avg_ag_hr']:.4f},{metrics['avg_ag_sr']:.4f},{metrics['avg_ag_up']:.4f}, "
                    f"{metrics['avg_rmse_sr']:.4f},{metrics['avg_rmse_up']:.4f},{metrics['avg_blur_sr']:4f}, {metrics['avg_blur_up']:.4f}\n")


def main():
    # Paths
    sr_path = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\result\delft4"
    hr_path = r"D:\Super_Resolution\Delft\HR\real_hr\tiles_256"
    up_root_path = r"D:\Super_Resolution\Delft\HR\generated_hr_normal_upscale\tiles_256"
    output_metrics_file = "category_metrics_results_new.csv"

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
        "BRISQUE(SR)", "BRISQUE(UP)", "AG(HR)", "AG(SR)", "AG(UP)", "RMSE(SR)", "RMSE(UP)", "BLUR(SR)", "BLUR(UP)"
    ]
    table_data = []

    for category, metrics in category_metrics.items():
        table_data.append([
            category,
            f"{metrics['avg_psnr_sr']:.4f}", f"{metrics['avg_psnr_up']:.4f}",
            f"{metrics['avg_ssim_sr']:.4f}", f"{metrics['avg_ssim_up']:.4f}",
            f"{metrics['avg_brisque_sr']:.4f}", f"{metrics['avg_brisque_up']:.4f}",
            f"{metrics['avg_ag_hr']:.4f}", f"{metrics['avg_ag_sr']:.4f}", f"{metrics['avg_ag_up']:.4f}",
            f"{metrics['avg_rmse_sr']:.4f}", f"{metrics['avg_rmse_up']:.4f}",
            f"{metrics['avg_blur_sr']:.4f}", f"{metrics['avg_blur_up']:.4f}"
        ])

    print("\nCategory-wise results:")
    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))

    # Print footnotes for metric interpretation
    print("\n" + "-" * 80)
    print("Metric Interpretation:")
    print("• PSNR: Higher is better. If PSNR(SR) > PSNR(UP), SR performs better than simple upscaling.")
    print("• SSIM: Higher means more perceptual similarity to ground truth. If SSIM(SR) > SSIM(UP), SR is preferable.")
    print("• BRISQUE: Lower is better. If BRISQUE(SR) < BRISQUE(UP), SR produces more natural images.")
    print("• RMSE: Lower is better. If RMSE(SR) < RMSE(UP), SR is closer to ground truth in terms of pixel accuracy.")
    print("• Average Gradient (AG): Higher means sharper edges. If AG(SR) > AG(UP), SR preserves textures better.")
    print(
        "• Blur Score: Lower values indicate sharper images. If BLUR(SR) < BLUR(UP), SR reduces blur more effectively.")
    print("-" * 80)


if __name__ == "__main__":
    main()