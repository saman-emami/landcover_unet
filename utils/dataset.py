import os
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torchvision
import albumentations as A
from typing import List, Optional, Tuple


class LandCoverDataset(Dataset):
    """
    PyTorch Dataset for the LandCover.ai semantic segmentation dataset.

    Each sample consists of an RGB aerial image and its corresponding segmentation mask.
    The dataset supports optional image/mask albumentations transformation.

    Parameters:
        image_dir (str): Directory containing input RGB images.
        mask_dir (str): Directory containing segmentation masks.
        image_list (List[str]): List of image filenames (should match between images and masks).
        num_classes (int): number of semantic classes (incl. background).
        transform (Optional[albumentations.Compose]):
            Optional albumentations transformation
    """

    def __init__(
        self,
        processed_data_dir: str,
        num_classes: int,
        transform: Optional[A.Compose] = None,
    ):
        self.processed_data_dir = processed_data_dir
        self.num_classes = num_classes
        self.transform = transform

        data = np.load(processed_data_dir)

        self.images = data["images"]
        self.masks = data["masks"]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and returns a single sample (image, mask) at the given index."""

        image = self.images[index]
        mask = self.masks[index]

        # If a transform is provided (albumentations), apply it to both image and mask
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image_tensor = torchvision.transforms.ToTensor()(image)
        mask_tensor = torch.from_numpy(mask).long()

        mask_tensor = (
            F.one_hot(mask_tensor, num_classes=self.num_classes)
            .permute(2, 0, 1)
            .float()
        )  # -> [C, H, W] float32

        return image_tensor, mask_tensor
