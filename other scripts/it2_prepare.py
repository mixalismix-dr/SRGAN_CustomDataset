import os
import shutil
import re

# Paths
lr_dir = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\custom_dataset\It2\train_LR6"
hr_source_dir = r"D:\Super_Resolution\Delft\HR\tiles_256x256"
hr_dest_dir = r"C:\Users\mike_\OneDrive\Desktop\MSc Geomatics\Master Thesis\Codebases\SRGAN_CustomDataset\custom_dataset\It2\train_HR6"

os.makedirs(hr_dest_dir, exist_ok=True)

# Pattern to extract tile number
pattern = re.compile(r"LR_tile_(\d+)\.tif")

for lr_file in os.listdir(lr_dir):
    match = pattern.match(lr_file)
    if match:
        tile_number = match.group(1)
        hr_filename = f"HR_tile_{tile_number}.tif"
        hr_path = os.path.join(hr_source_dir, hr_filename)
        dest_path = os.path.join(hr_dest_dir, hr_filename)

        if os.path.exists(hr_path):
            shutil.copy(hr_path, dest_path)
            print(f"Copied: {hr_filename}")
        else:
            print(f"Missing HR tile for: {tile_number}")
