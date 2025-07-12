import argparse
from ..config import Config
import albumentations as A


def pars_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand the landcover.ai.v1 training set"
    )
    parser.add_argument(
        "--input-root",
        help="Folder with 'images/' and 'masks/' subdirs",
        type=str,
        default=Config.RAW_DIR,
    )
    parser.add_argument(
        "--output-root",
        help="Destination root for augmented dataset",
        type=str,
        default=Config.PROCESSED_DIR,
    )
    parser.add_argument(
        "--repaeat",
        type=int,
        help="How many *extra* augmentations per original",
        default=10,
    )

    parser.add_argument(
        "--img-size",
        type=int,
        help="Resize size used in pipeline",
        default=Config.IMG_SIZE,
    )

    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility", default=42
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = pars_args()
