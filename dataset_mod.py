
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import random

class mydata(Dataset):
    def __init__(self, LR_path, GT_path, mask_path, in_memory=True, transform=None):
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.mask_path = mask_path  # Path to the directory with mask tiles
        self.in_memory = in_memory
        self.transform = transform

        # List of image files in the directories
        self.LR_img = sorted(os.listdir(LR_path))
        self.GT_img = sorted(os.listdir(GT_path))
        self.mask_img = sorted(os.listdir(mask_path))  # List of mask tile files

        # Debugging: Ensure the mask directory contains the correct files
        # print(f"Mask tiles found: {len(self.mask_img)}")

    def __len__(self):
        return len(self.LR_img)

    def __getitem__(self, i):
        img_item = {}

        # Load the LR and GT images
        LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i])).convert("RGB")).astype(np.float32)
        GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i])).convert("RGB")).astype(np.float32)

        # Extract the name of the current tile (without the extra suffix)
        mask_tile_name = self.LR_img[i].replace('_lanczos_down', '')  # Remove the suffix to match the mask tile
        mask_tile_path = os.path.join(self.mask_path, mask_tile_name)

        # Debugging: Print the path of the mask tile being loaded
        # print(f"Loading mask for: {mask_tile_name} from {mask_tile_path}")

        # Check if the mask tile exists
        if not os.path.exists(mask_tile_path):
            raise FileNotFoundError(f"Mask tile {mask_tile_name} not found in {self.mask_path}")

        # Load the corresponding mask tile
        mask = np.array(Image.open(mask_tile_path).convert("L")).astype(np.float32)  # Shape: (256, 256)


        # # Check the values in the mask
        # mask_min = mask.min()
        # mask_max = mask.max()
        # print(mask_min, mask_max)

        # Resize the mask to match LR dimensions (64x64)
        mask_lr = np.array(Image.fromarray(mask).resize((64, 64), Image.BILINEAR))  # Shape: (64, 64)

        # Add a channel dimension to the masks
        mask_hr = np.expand_dims(mask, axis=-1)  # Shape: (256, 256, 1)
        mask_lr = np.expand_dims(mask_lr, axis=-1)  # Shape: (64, 64, 1)

        LR = np.concatenate((LR, mask_lr), axis=-1)
        GT = np.concatenate((GT, mask_hr), axis=-1)


        # print("MASKS SHAPE", mask_hr.shape, mask_lr.shape)
        # print("GT LR Shapes", GT.shape, LR.shape)




        # Normalize the images and the masks
        img_item['GT'] = (GT / 127.5) - 1.0  # Normalize to [-1, 1]
        img_item['LR'] = (LR / 127.5) - 1.0  # Normalize to [-1, 1]
        img_item['mask_hr'] = mask_hr  # Normalize to [-1, 1]
        img_item['mask_lr'] = mask_lr # Normalize to [-1, 1]
        # print(img_item)
        # print(mask.min(), mask.max())
        # print(mask_lr.min(), mask_lr.max())
        # print(img_item.keys())

        # Apply any transformations if provided
        if self.transform is not None:
            img_item = self.transform(img_item)

        # Ensure the images and masks are in the correct shape for input
        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)  # Shape: (3, 256, 256)
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)  # Shape: (3, 64, 64)
        img_item['mask_hr'] = img_item['mask_hr'].transpose(2, 0, 1).astype(np.float32)  # Shape: (1, 256, 256)
        img_item['mask_lr'] = img_item['mask_lr'].transpose(2, 0, 1).astype(np.float32)  # Shape: (1, 64, 64)


        return img_item


class testOnly_data(Dataset):
    def __init__(self, LR_path, mask_path, in_memory=True, transform=None):
        self.LR_path = LR_path
        self.mask_path = mask_path  # Path to the mask directory
        self.LR_img = sorted(os.listdir(LR_path))
        self.mask_img = sorted(os.listdir(mask_path))  # List of mask files
        self.in_memory = in_memory

        # Optionally load images into memory if in_memory is True
        if in_memory:
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr))) for lr in self.LR_img]
            self.mask_img = [np.array(Image.open(os.path.join(self.mask_path, mask))) for mask in self.mask_img]

    def __len__(self):
        return len(self.LR_img)

    def __getitem__(self, i):
        img_item = {}

        if self.in_memory:
            LR = self.LR_img[i]
            mask = self.mask_img[i]
        else:
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i])))
            mask = np.array(Image.open(os.path.join(self.mask_path, self.mask_img[i])))

        # Ensure LR is a 3-channel image (H, W, C)
        if len(LR.shape) == 2:  # If grayscale, convert to 3 channels
            LR = np.stack([LR] * 3, axis=-1)

        # Ensure mask is a single-channel image (H, W, 1)
        if len(mask.shape) == 3:  # If mask has 3 channels, convert to grayscale
            mask = mask.mean(axis=-1, keepdims=True)  # Convert to grayscale (H, W, 1)
        elif len(mask.shape) == 2:  # If mask is 2D, add a channel dimension
            mask = np.expand_dims(mask, axis=-1)  # Shape: (H, W, 1)

        # Normalize the LR image to [-1, 1]
        img_item['LR'] = (LR / 127.5) - 1.0  # Normalize to [-1, 1]
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)  # Convert to (C, H, W)

        # Normalize the mask to [-1, 1]
        img_item['mask_lr'] = (mask / 127.5) - 1.0  # Normalize to [-1, 1]
        img_item['mask_lr'] = img_item['mask_lr'].transpose(2, 0, 1).astype(np.float32)  # Convert to (1, H, W)

        return img_item


class crop(object):
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size

    def __call__(self, sample):
        LR_img, GT_img, mask_hr, mask_lr = sample['LR'], sample['GT'], sample['mask_hr'], sample['mask_lr']
        ih, iw = LR_img.shape[:2]

        ix = random.randrange(0, iw - self.patch_size + 1)
        iy = random.randrange(0, ih - self.patch_size + 1)

        tx = ix * self.scale
        ty = iy * self.scale

        LR_patch = LR_img[iy: iy + self.patch_size, ix: ix + self.patch_size]
        GT_patch = GT_img[ty: ty + (self.scale * self.patch_size), tx: tx + (self.scale * self.patch_size)]
        mask_hr_patch = mask_hr[ty: ty + (self.scale * self.patch_size), tx: tx + (self.scale * self.patch_size)]
        mask_lr_patch = mask_lr[iy: iy + self.patch_size, ix: ix + self.patch_size]

        return {'LR': LR_patch, 'GT': GT_patch, 'mask_hr': mask_hr_patch, 'mask_lr': mask_lr_patch}

class augmentation(object):
    def __call__(self, sample):
        LR_img, GT_img, mask_hr, mask_lr = sample['LR'], sample['GT'], sample['mask_hr'], sample['mask_lr']

        hor_flip = random.randrange(0, 2)
        ver_flip = random.randrange(0, 2)
        rot = random.randrange(0, 2)

        if hor_flip:
            LR_img = np.fliplr(LR_img)
            GT_img = np.fliplr(GT_img)
            mask_hr = np.fliplr(mask_hr)
            mask_lr = np.fliplr(mask_lr)

        if ver_flip:
            LR_img = np.flipud(LR_img)
            GT_img = np.flipud(GT_img)
            mask_hr = np.flipud(mask_hr)
            mask_lr = np.flipud(mask_lr)

        if rot:
            LR_img = LR_img.transpose(1, 0, 2)
            GT_img = GT_img.transpose(1, 0, 2)
            mask_hr = mask_hr.transpose(1, 0, 2)
            mask_lr = mask_lr.transpose(1, 0, 2)

        return {'LR': LR_img, 'GT': GT_img, 'mask_hr': mask_hr, 'mask_lr': mask_lr}


