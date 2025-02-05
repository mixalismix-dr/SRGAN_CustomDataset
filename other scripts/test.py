import os
import cv2
from tqdm import tqdm

# Set the directory containing images
image_dir = r"/custom_dataset/train_LR3"  # Change this path

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".tif"))]

# Check each image
for img_file in tqdm(image_files, desc="Checking images"):
    img_path = os.path.join(image_dir, img_file)

    # Try loading the image
    img = cv2.imread(img_path)

    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        print(f"❌ Corrupted or empty image: {img_path}")
    else:
        print(f"✅ Valid image: {img_path}")
