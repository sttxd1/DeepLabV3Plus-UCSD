# dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import json

class UCSD(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, labels_json='labels.json'):
        """
        Args:
            images_dir (str): Directory with all the images.
            labels_dir (str): Directory with all the labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            labels_json (str): Path to the JSON file containing color-class mappings.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Load color to class mapping from JSON
        self.color_to_class = self.load_color_mapping(labels_json)

        # Get list of image and label files
        self.images = sorted(os.listdir(self.images_dir))
        self.labels = sorted(os.listdir(self.labels_dir))

        assert len(self.images) == len(self.labels), "Number of images and labels should be equal."

        # Optionally, ensure that image and label filenames correspond
        for img_file, label_file in zip(self.images, self.labels):
            assert os.path.splitext(img_file)[0] == os.path.splitext(label_file)[0], \
                f"Image and label filenames do not match: {img_file} vs {label_file}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        image_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.labels[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image, label = self.transform(image, label)

        # Convert label from RGB to class indices
        label = self.rgb_to_class(label)

        return image, label

    def load_color_mapping(self, json_file):
        # Load color-class mapping from the JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)

        color_to_class = {}
        idx = 0  # Start class indices from 0

        for label_info in data['labels']:
            if label_info.get('evaluate', False):
                color = tuple(label_info['color'])  # Convert color list to tuple
                color_to_class[color] = idx
                idx += 1

        return color_to_class

    def rgb_to_class(self, label):
        # Convert PIL Image to NumPy array
        label_np = np.array(label)
        height, width, _ = label_np.shape
        label_class = np.zeros((height, width), dtype=np.int64)

        # Map each pixel to its corresponding class index
        for color, class_idx in self.color_to_class.items():
            matches = np.all(label_np == color, axis=-1)
            label_class[matches] = class_idx

        # Handle pixels with colors not in color_to_class
        # Optionally set them to a special index, e.g., 255 for 'ignore'
        # unmatched_pixels = ~np.isin(label_np.reshape(-1, 3), list(self.color_to_class.keys())).all(axis=1)
        # label_class = label_class.reshape(-1)
        # label_class[unmatched_pixels] = 255  # Or any other value you choose
        # label_class = label_class.reshape(height, width)

        # Convert to PyTorch tensor
        label_class = torch.from_numpy(label_class)

        return label_class

train_images_dir = '/lich-central/dataset/ucsd/training/images/'
train_labels_dir = '/lich-central/dataset/ucsd/training/v1.2/labels/'
val_images_dir = '/lich-central/dataset/ucsd/validation/images/'
val_labels_dir = '/lich-central/dataset/ucsd/validation/v1.2/labels'


