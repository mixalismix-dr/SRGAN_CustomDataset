import os
import cv2
import numpy as np
import scipy.io
from tqdm import tqdm
from skimage.color import rgb2ycbcr
import scipy.ndimage

# Path to HR dataset (256x256, 8cm resolution)
hr_image_path = r"D:\Super_Resolution\Delft\HR\real_hr\tiles_256"


# Function to compute MSCN coefficients
def compute_mscn(image, kernel_size=7, sigma=7.0 / 6.0):
    window = cv2.getGaussianKernel(kernel_size, sigma)
    window = np.outer(window, window.transpose())

    mu = cv2.filter2D(image, -1, window, borderType=cv2.BORDER_REFLECT)
    mu_sq = mu ** 2
    sigma = np.sqrt(cv2.filter2D(image ** 2, -1, window, borderType=cv2.BORDER_REFLECT) - mu_sq)

    return (image - mu) / (sigma + 1e-7), sigma, mu


# Extract paired products
def paired_products(mscn):
    shift1 = np.roll(mscn, 1, axis=1)
    shift2 = np.roll(mscn, 1, axis=0)
    shift3 = np.roll(np.roll(mscn, 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(mscn, 1, axis=0), -1, axis=1)

    return shift1 * mscn, shift2 * mscn, shift3 * mscn, shift4 * mscn


# Feature extraction function
def extract_features(img_y):
    # Compute MSCN coefficients
    mscn, _, _ = compute_mscn(img_y)

    # Compute paired products
    h_img, v_img, d1_img, d2_img = paired_products(mscn)

    # Compute basic NSS features
    features = [
        np.mean(mscn), np.std(mscn),  # MSCN Mean and STD
        np.mean(h_img), np.std(h_img),
        np.mean(v_img), np.std(v_img),
        np.mean(d1_img), np.std(d1_img),
        np.mean(d2_img), np.std(d2_img),
        np.mean(mscn ** 2), np.std(mscn ** 2),  # Higher-order statistics
        np.mean(np.abs(mscn)), np.std(np.abs(mscn)),  # Absolute value statistics
        np.mean(mscn ** 3), np.std(mscn ** 3),  # Skewness
        np.mean(mscn ** 4), np.std(mscn ** 4),  # Kurtosis
    ]

    return np.array(features)


# List all image files
hr_images = [os.path.join(hr_image_path, img) for img in os.listdir(hr_image_path) if
             img.endswith(('.jpg', '.png', '.tif'))]

# Container for extracted features
features_list = []

# Extract features for each HR image
for img_path in tqdm(hr_images, desc="Extracting NIQE features"):
    img = cv2.imread(img_path)
    if img is None:
        continue  # Skip unreadable files

    # Convert to grayscale (Y-channel of YCbCr)
    img_y = rgb2ycbcr(img)[:, :, 0]

    # Normalize pixel values to [0, 1]
    img_y = img_y.astype(np.float32) / 255.0

    # Compute NIQE features
    features = extract_features(img_y)
    features_list.append(features)

# Convert to NumPy array
features_array = np.array(features_list)

# Compute mean and covariance of the feature vectors
mu_prisparam = np.mean(features_array, axis=0)  # Mean feature vector
cov_prisparam = np.cov(features_array, rowvar=False)  # Covariance matrix

# Save to modelparameters.mat in the correct format
model_params = {
    "pop_mu": mu_prisparam,  # This is expected by NIQE
    "pop_cov": cov_prisparam  # This is expected by NIQE
}

scipy.io.savemat("modelparameters.mat", model_params)
print("✅ Custom NIQE model trained and saved as modelparameters.mat")
