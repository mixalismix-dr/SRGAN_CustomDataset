import os
import shutil
import re

# Define the base directories for HR and LR tiles
hr_base_dir = r"D:\Super_Resolution\Utrecht\iteration_2\tiles_256x256"
lr_base_dir = r"D:\Super_Resolution\Utrecht\iteration_2\tiles_64x64"

# Function to extract the tile number from the filename
def extract_tile_number(filename):
    match = re.search(r'HR_tile_(\d+)', filename)
    return match.group(1) if match else None

# Traverse through each category subfolder in the HR directory
for category in os.listdir(hr_base_dir):
    hr_category_path = os.path.join(hr_base_dir, category)
    lr_category_path = os.path.join(lr_base_dir, category)

    # Ensure the path is a directory
    if not os.path.isdir(hr_category_path):
        continue

    # Create the corresponding category subfolder in the LR directory if it doesn't exist
    os.makedirs(lr_category_path, exist_ok=True)

    # Iterate over the HR tiles in the current category subfolder
    for hr_tile in os.listdir(hr_category_path):
        tile_number = extract_tile_number(hr_tile)
        if tile_number:
            lr_tile_name = f"LR_tile_{tile_number}.tif"
            lr_tile_path = os.path.join(lr_base_dir, lr_tile_name)
            if os.path.exists(lr_tile_path):
                shutil.copy(lr_tile_path, lr_category_path)
                print(f"Copied {lr_tile_name} to {lr_category_path}")
            else:
                print(f"LR tile {lr_tile_name} not found in {lr_base_dir}")
