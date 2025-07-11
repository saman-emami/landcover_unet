import albumentations as A

imagenet_normalization = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


def get_train_transforms(
    image_size: int = 512,
) -> A.Compose:
    """Training augmentations"""

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.ElasticTransform(p=0.3),
            A.Normalize(
                mean=imagenet_normalization["mean"],
                std=imagenet_normalization["std"],
            ),
        ],
        additional_targets={"mask": "mask"},  # This enables synchronized augmentation,
    )


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """Validation transforms (no augmentation)"""

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=imagenet_normalization["mean"],
                std=imagenet_normalization["std"],
            ),
        ],
        additional_targets={"mask": "mask"},
    )
