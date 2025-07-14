import numpy as np
from typing import Optional
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from ..configs.config import Config
from typing import Tuple


class AugmentDatsetArgs(argparse.Namespace):
    raw_dir: Path
    processed_dir: Path
    patch_size: int
    stride: int


def read_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int], bool]:
    """
    Reads an image from the specified file path using OpenCV and determines if it is a mask.

    Parameters
    ----------
    image_path: str
        The file path to the image to be loaded.

    Returns
    -------
    image: np.ndarray
        The loaded image as a NumPy array.

    shape: Tuple[int, int]
        The shape of the image as (height, width).

    is_mask: bool
        True if the image is considered a mask (2-dimensional), False otherwise.

    Raises
    ------
    ValueError:
        If the image at `image_path` doesn't exist.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")

    is_mask = image.ndim == 2

    shape = image.shape[:2]

    return (image, shape, is_mask)


def slice_image(
    image_path: str,
    patch_size: int,
    save_path: str,
    stride: int,
) -> None:
    """
    Slices an the image at `image_path` to smaller patches of size`patch_size`
    with the given `stride`. Saves the resulting patches to `save_path`

    Parameters
    ----------
    image_path: str
        The file path to the image to be loaded.

    patch_size: int
        Size of the sliced patches.

    save_path: str
        Path to where the pathces should be saved.

    stride: int
        Stride of the sliding window for slicing.
    """

    image, shape, is_mask = read_image(image_path)

    height, width = shape

    for y in range(0, height, stride):
        for x in range(0, width, stride):

            if y + patch_size > height or x + patch_size > width:
                continue

            if is_mask:
                patch = image[y : y + patch_size, x : x + patch_size]
            else:
                patch = image[y : y + patch_size, x : x + patch_size, :]

            cv2.imwrite(save_path, patch)


def process_raw_dataset(
    raw_dir: Path,
    processed_dir: Path,
    patch_size: int,
    stride: Optional[int],
):
    """
    processes the entire dataset at `raw_dir` and saves the processed images to `processed_dir`.

    Parameters
    ----------
    raw_dir: Path
        Path to the raw data containing **/images** and **/masks** directories.

    processed_dir:
        Path to where the processed data should be saved.

    patch_size: int
        Size of the sliced patches.

    stride: Optional[int] = None
        Stride of the sliding window for slicing. internally defaults to `patch_size`.

    Raises
    ------
    ValueError:
        If no images are found at raw_dir/images or raw_dir/masks

    """

    raw_images_dir = raw_dir / "images"
    raw_mask_dir = raw_dir / "masks"

    if stride is None:
        stride = patch_size

    image_paths = sorted(raw_images_dir.glob("*.tif"))
    mask_paths = sorted(raw_mask_dir.glob("*.tif"))

    if not image_paths:
        raise ValueError(f"No .tif images found in {raw_images_dir}")

    if not mask_paths:
        raise ValueError(f"No .tif masks found in {raw_mask_dir}")

    for image_path, mask_path in tqdm(
        zip(image_paths, mask_paths), desc="Processing dataset"
    ):
        if image_path.name != mask_path.name:
            ValueError("There is a mismatch in the images and masks directories.")

        slice_image(
            image_path=str(image_path),
            patch_size=patch_size,
            save_path=str(processed_dir / "images"),
            stride=stride,
        )
        slice_image(
            image_path=str(mask_path),
            patch_size=patch_size,
            save_path=str(processed_dir / "masks"),
            stride=stride,
        )


def main(args: AugmentDatsetArgs) -> None:
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    process_raw_dataset(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        patch_size=args.patch_size,
        stride=args.stride,
    )


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for slicing the LandCover.ai dataset.

    Returns:
        argparse.ArgumentParser: Configured parser object.
    """
    parser = argparse.ArgumentParser(
        description="Slice LandCover.ai dataset into patches"
    )
    parser.add_argument(
        "--raw-dir",
        default=Config.RAW_DIR,
        type=Path,
        help="Root directory with images/ and masks/",
    )
    parser.add_argument(
        "--processed-dir",
        default=Config.PROCESSED_DIR,
        type=Path,
        help="Output root for sliced patches",
    )
    parser.add_argument(
        "--patch-size",
        default=Config.PATCH_SIZE,
        type=int,
        help="Patch dimension (square)",
    )
    parser.add_argument(
        "--stride",
        default=Config.SLICING_STRIDE,
        type=int,
        help="Stride between patches default",
    )

    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args(namespace=AugmentDatsetArgs())
    main(args)
