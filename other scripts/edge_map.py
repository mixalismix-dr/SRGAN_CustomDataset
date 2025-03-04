import os
import glob
import cv2
import numpy as np

sr_path = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\result\delft4"
hr_path = r"D:\Super_Resolution\Delft\HR\real_hr\tiles_256"
up_path = r"D:\Super_Resolution\Delft\HR\generated_hr_normal_upscale\tiles_256"
output_dir = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\edge_maps"

os.makedirs(output_dir, exist_ok=True)

sr_images = sorted(glob.glob(os.path.join(sr_path, "res_*_down.tif")))[:5]

for idx, sr_img_path in enumerate(sr_images):
    sr_filename = os.path.basename(sr_img_path)
    tile_name = sr_filename.replace("res_", "").replace("_down.tif", "")

    hr_img_path = os.path.join(hr_path, f"{tile_name}.tif")
    up_candidates = glob.glob(os.path.join(up_path, "**", f"{tile_name}_down_up.tif"), recursive=True)

    if not os.path.exists(hr_img_path) or not up_candidates:
        print(f"Skipping {tile_name} (missing HR or UP image)")
        continue

    up_img_path = up_candidates[0]

    sr_image = cv2.imread(sr_img_path)
    hr_image = cv2.imread(hr_img_path)
    up_image = cv2.imread(up_img_path)

    if sr_image is None or hr_image is None or up_image is None:
        print(f"Skipping {tile_name} (unable to read one or more images)")
        continue

    sr_gray = cv2.cvtColor(sr_image, cv2.COLOR_BGR2GRAY)
    hr_gray = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)
    up_gray = cv2.cvtColor(up_image, cv2.COLOR_BGR2GRAY)

    edges_sr = cv2.Canny(sr_gray, 100, 200)
    edges_hr = cv2.Canny(hr_gray, 100, 200)
    edges_up = cv2.Canny(up_gray, 100, 200)

    cv2.imwrite(os.path.join(output_dir, f"{idx:02d}_{tile_name}_edges_sr.png"), edges_sr)
    cv2.imwrite(os.path.join(output_dir, f"{idx:02d}_{tile_name}_edges_hr.png"), edges_hr)
    cv2.imwrite(os.path.join(output_dir, f"{idx:02d}_{tile_name}_edges_up.png"), edges_up)

    print(f"Saved edge maps for {tile_name}")
