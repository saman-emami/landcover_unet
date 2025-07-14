import argparse
import zipfile
import os
from tqdm import tqdm
from configs.config import Config


class ExtractDatasetArgs(argparse.Namespace):
    raw_dir: str
    dataset_file_name: str
    data_dir: str


def extract_dataset(raw_dir: str, dataset_file_name: str, data_dir: str) -> None:
    """
    Extract the LandCover.ai dataset zip file into the specified directory.

    If the extraction directory already exists skips the extraction. Otherwise
    extracts all contents from the zip file into the raw data directory.

    Parameters
    ----------
    raw_dir : str
        Target directory to extract the dataset into.

    dataset_file_name : str
        Name of the dataset zip file to extract.

    data_dir : str
        Directory containing the dataset zip file.
    """

    dataset_already_extracted = os.path.exists(raw_dir)
    if dataset_already_extracted:
        print("Dataset has already been extracted.")
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
        description="Extract the LandCover.ai dataset zip file."
    )
    parser.add_argument(
        "--raw-dir",
        default=Config.RAW_DIR,
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
