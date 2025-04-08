import os
import rasterio

lr_dir = "/mnt/SRGAN/custom_dataset/It1/train_LR4"
hr_dir = "/mnt/SRGAN/custom_dataset/It1/train_HR4"

lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.tif')])
hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.tif')])

# 1. Check count match
print(f"LR count: {len(lr_files)}, HR count: {len(hr_files)}")
if len(lr_files) != len(hr_files):
    print("[WARNING] Different number of files in LR and HR folders.")

# 2. Check dimensions
for lr_file, hr_file in zip(lr_files, hr_files):
    lr_path = os.path.join(lr_dir, lr_file)
    hr_path = os.path.join(hr_dir, hr_file)

    with rasterio.open(lr_path) as src_lr:
        lr_shape = (src_lr.height, src_lr.width)
    with rasterio.open(hr_path) as src_hr:
        hr_shape = (src_hr.height, src_hr.width)

    if lr_shape != (64, 64):
        print(f"[LR] {lr_file} is {lr_shape}, expected (64, 64)")
    if hr_shape != (256, 256):
        print(f"[HR] {hr_file} is {hr_shape}, expected (256, 256)")
