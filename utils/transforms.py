import albumentations as A

imagenet_normalization = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


def get_train_transforms() -> A.Compose:
    """
    Create the Albumentations augmentation pipeline for training images and masks.

    Returns
    -------
    A.Compose
        Albumentations Compose object containing the training augmentations.
        The returned transform expects input as:
            - image: np.ndarray (H, W, 3), dtype uint8 or float32
            - mask:  np.ndarray (H, W),    dtype int or uint8
        and returns a dict with transformed 'image' and 'mask'.
    """
    return A.Compose(
        [
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


def get_val_transforms() -> A.Compose:
    """
    Create the Albumentations transformation pipeline for validation images and masks.

    This function returns a composition that only normalizes the input image
    (no augmentation).

    Returns
    -------
    A.Compose
        Albumentations Compose object containing the validation transforms.
        The returned transform expects input as:
            - image: np.ndarray (H, W, 3), dtype uint8 or float32
            - mask:  np.ndarray (H, W),    dtype int or uint8
        and returns a dict with transformed 'image' and 'mask'.
    """

    return A.Compose(
        [
            A.Normalize(
                mean=imagenet_normalization["mean"],
                std=imagenet_normalization["std"],
            ),
        ],
        additional_targets={"mask": "mask"},
    )
