import argparse
import zipfile
import os
from tqdm import tqdm
from ..configs.config import Config


class ExtractDatasetArgs(argparse.Namespace):
    raw_dir: str
    dataset_file_name: str
    data_dir: str


def extract_dataset(raw_dir: str, dataset_file_name: str, data_dir: str) -> None:
    dataset_already_extracted = os.path.exists(raw_dir)
    if dataset_already_extracted:
        return

    zip_path = os.path.join(data_dir, dataset_file_name)
    with zipfile.ZipFile(zip_path) as zf, tqdm(
        total=len(zf.infolist()), desc="Extracting"
    ) as bar:
        for member in zf.infolist():
            zf.extract(member, raw_dir)
            bar.update(1)


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for extracting the LandCover.ai dataset zip file.

    Returns:
        argparse.ArgumentParser: Configured parser object.
    """
    parser = argparse.ArgumentParser(
        description="Slice LandCover.ai dataset into patches"
    )
    parser.add_argument(
        "--raw-dir",
        default=Config.DATA_DIR,
        type=str,
        help="Where the raw dataset should be extracted into.",
    )
    parser.add_argument(
        "--dataset-file-name",
        default=Config.DATASET_FILE_NAME,
        type=str,
        help="The name of downloaded dataset zip file.",
    )
    parser.add_argument(
        "--data-dir",
        default=Config.DATA_DIR,
        type=str,
        help="The directory the downloaded file should be read from.",
    )

    return parser


def main(args: ExtractDatasetArgs) -> None:
    extract_dataset(
        raw_dir=args.raw_dir,
        dataset_file_name=args.dataset_file_name,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args(namespace=ExtractDatasetArgs())
    main(args)
