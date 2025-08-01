import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import albumentations as A
from typing import List, Optional, Tuple


class LandCoverDataset(Dataset):
    """
    PyTorch Dataset for the LandCover.ai semantic segmentation dataset.

    Each sample consists of an RGB aerial image and its corresponding segmentation mask.
    The dataset supports optional image/mask albumentations transformation.

    Parameters
    ----------
    image_dir: str
        Directory containing input RGB images.

    mask_dir: str
        Directory containing segmentation masks.

    image_list: List[str]
        List of image filenames (should match between images and masks).

    num_classes: int
        number of semantic classes (incl. background).

    transform: Optional[albumentations.Compose]
        Optional albumentations transformation
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_list: List[str],
        num_classes: int,
        transform: Optional[A.Compose] = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the image filename for this index
        image_name = self.image_list[index]

        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        mask_path = os.path.join(self.mask_dir, image_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

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
