"""
Dataset handling and preprocessing.
This script provides utilities for loading, processing, and transforming images and masks.
"""

# Import necessary libraries
import torch
import os
import cv2
from monai.data import DataLoader, Dataset, CacheDataset
import nibabel as nib
import config
import numpy as np

# Define a function to read an image from a given path
def load_grayscale_image(image_path):
    """
    Reads a grayscale image from the given file path, resizes it to the configured size,
    normalizes pixel values to the range [0,1], and returns the processed image.
    """
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (config.img_size, config.img_size), interpolation=cv2.INTER_CUBIC)
    image = image / image.max()
    return image

# Define a function to read a segmentation mask from a given path
def load_segmentation_mask(mask_path):
    """
    Reads a segmentation mask from the given file path, resizes it to the configured size,
    and maps pixel values to predefined class labels.
    """
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (config.img_size, config.img_size), interpolation=cv2.INTER_NEAREST)
    mask[mask == 0] = 0
    mask[mask == 100] = 1
    mask[mask == 150] = 2
    mask[mask == 200] = 3
    return mask

# Custom dataset class for loading image-mask pairs
class CardiacDataset(Dataset):
    def __init__(self, image_paths, mask_paths, num_mid_frames=None, transform=None):
        """
        Initializes the dataset with image and mask file paths.

        Args:
            image_paths (list): List of file paths for images.
            mask_paths (list): List of file paths for masks.
            num_mid_frames (int, optional): Number of intermediate frames to include.
            transform (callable, optional): Transformation function to apply to data.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_mid_frames = num_mid_frames
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def _load_image_and_mask(self, image_base_path, mask_base_path, frame_type):
        """
        Helper function to load an image and its corresponding mask.

        Args:
            image_base_path (str): Base path for the image file.
            mask_base_path (str): Base path for the mask file.
            frame_type (str): Type of frame to load (e.g., 'ED', 'ES', 'Mid1').
        
        Returns:
            dict: Dictionary containing the image, mask, and filename.
        """
        image = load_grayscale_image(f"{image_base_path}{frame_type}.png")
        mask = load_segmentation_mask(f"{mask_base_path}{frame_type}.png")
        return {'image': image, 'mask': mask, 'name': os.path.basename(image_base_path)}

    def __getitem__(self, index):
        """
        Retrieves the image-mask pair for a given index, including ED and ES frames.
        If specified, also loads intermediate frames.

        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            dict: Dictionary containing ED, ES, and (optionally) mid-frame images and masks.
        """
        image_directory, image_filename = os.path.split(self.image_paths[index])
        mask_directory, _ = os.path.split(self.mask_paths[index])

        image_base_path = os.path.join(image_directory, image_filename[:-6])
        mask_base_path = os.path.join(mask_directory, image_filename[:-6])

        # Load ED and ES images and masks
        ed_frame = self._load_image_and_mask(image_base_path, mask_base_path, 'ED')
        es_frame = self._load_image_and_mask(image_base_path, mask_base_path, 'ES')

        # Load mid-frame images and masks if specified
        mid_frames = {}
        if self.num_mid_frames is not None:
            for i in range(1, self.num_mid_frames + 1):
                mid_frames[f'Mid{i}'] = self._load_image_and_mask(image_base_path, mask_base_path, f'Mid{i}')

        # Apply transformations or convert to tensor
        def apply_transformations(data_sample):
            if self.transform:
                return self.transform(data_sample)
            else:
                data_sample['image'] = torch.tensor(data_sample['image'], dtype=torch.float32).unsqueeze(0)
                data_sample['mask'] = torch.tensor(data_sample['mask'], dtype=torch.float32).unsqueeze(0)
                return data_sample

        ed_frame = apply_transformations(ed_frame)
        es_frame = apply_transformations(es_frame)

        if self.num_mid_frames is not None:
            for key in mid_frames:
                mid_frames[key] = apply_transformations(mid_frames[key])

        # Combine all frames into a single dictionary
        combined_data = {'ED': ed_frame, 'ES': es_frame}
        if self.num_mid_frames is not None:
            combined_data.update(mid_frames)

        return combined_data
