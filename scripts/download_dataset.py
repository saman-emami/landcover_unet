import os
import argparse
import requests
from tqdm import tqdm
from ..configs.config import Config


class DownloadDatasetArgs(argparse.Namespace):
    dataset_url: str
    dataset_file_name: str
    data_dir: str


def download_dataset(data_dir: str, dataset_file_name: str, dataset_url: str) -> None:
    """Download the LandCover.ai dataset

    Parameters
    ----------
    data_dir:
        The directory the downloaded file should be saved to.

    dataset_file_name:
        The name of the dataset zip file.

    dateset_url:
        The URL for downloading the dataset from.

    """
    zip_path = os.path.join(data_dir, dataset_file_name)
    dataset_already_downloaded = os.path.exists(zip_path)
    if dataset_already_downloaded:
        print("Dataset is already downloaded.")
        return

    with requests.get(dataset_url, stream=True, timeout=30) as resp:
        os.makedirs(data_dir, exist_ok=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 14):  # 16 KB
                f.write(chunk)
                bar.update(len(chunk))


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for downloading the LandCover.ai dataset.

    Returnsdataset_url
    -------
    argparse.ArgumentParser: Configured parser object.
    """
    parser = argparse.ArgumentParser(description="Download the LandCover.ai dataset.")

    parser.add_argument(
        "--dataset-url",
        default=Config.DATASET_URL,
        type=str,
        help="The URL for downloading the dataset.",
    )
    parser.add_argument(
        "--dataset-file-name",
        default=Config.DATASET_FILE_NAME,
        type=str,
        help="The name used to save the downloaded dataset zip file.",
    )
    parser.add_argument(
        "--data-dir",
        default=Config.DATA_DIR,
        type=str,
        help="The directory the downloaded file should be saved to.",
    )

    return parser


def main(args: DownloadDatasetArgs) -> None:
    download_dataset(
        data_dir=args.data_dir,
        dataset_url=args.dataset_url,
        dataset_file_name=args.dataset_file_name,
    )


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args(namespace=DownloadDatasetArgs())
    main(args)
