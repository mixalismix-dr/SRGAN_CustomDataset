import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob

# Paths
tiles_dir = r"D:\Super_Resolution\Delft\HR\real_hr\tiles_256\test"  # Parent folder containing category subfolders
image_size = (256, 256)  # Resize images to this size

# Define the categories (subdirectories)
categories = [d for d in os.listdir(tiles_dir) if os.path.isdir(os.path.join(tiles_dir, d))]
print(f"Found categories: {categories}")

# Define transformation
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])


class TileDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_samples_per_class=100):
        self.root_dir = root_dir
        self.transform = transform
        self.num_samples_per_class = num_samples_per_class
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {category: i for i, category in enumerate(categories)}

        # Load an equal number of images from each category
        for category in categories:
            category_path = os.path.join(root_dir, category)
            images = glob(os.path.join(category_path, "*.tif"))

            # Shuffle and take a subset
            random.shuffle(images)
            selected_images = images[:num_samples_per_class]

            self.image_paths.extend(selected_images)
            self.labels.extend([self.class_to_idx[category]] * len(selected_images))

        print(f"Loaded {len(self.image_paths)} images from {len(categories)} categories.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Create Dataset and DataLoader
dataset = TileDataset(root_dir=tiles_dir, transform=transform, num_samples_per_class=100)  # Adjust to 100 per category
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Example usage
for images, labels in dataloader:
    print(f"Batch: {images.shape}, Labels: {labels}")
    break  # Display only first batch
