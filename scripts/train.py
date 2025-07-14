import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any, Dict

from configs.config import Config
from models.unet import UNet
from models.losses import CombinedLoss
from utils.dataset import LandCoverDataset
from utils.metrics import SegmentationMetrics
from utils.transforms import get_train_transforms, get_val_transforms


class TrainingArgs(argparse.Namespace):
    processed_dir: str
    train_split: float
    val_split: float
    test_split: float
    num_classes: int
    batch_size: int
    input_channels: int
    learning_rate: float
    weight_decay: float
    num_epochs: int


def prepare_data_splits(
    processed_dir: str, train_split: float, val_split: float, test_split: float
) -> Tuple[List[str], List[str], List[str]]:
    """
    Prepare train/val/test splits

    Parameters
    ----------
    processed_dir: str
        Where the processed dataset should be read from.

    train_split: float
        The training fraction of the dataset (between 1 and).

    val_split: float
        The validation fraction of the dataset (between 1 and 0).

    test_split: float
        The testing fraction of the dataset (between 1 and 0).


    Returns
    -------
    train_split: List[str]
        Training portion of the dataset.

    val_split: List[str]
        Validation portion of the dataset.

    test_split: List[str]
        testing portion of the dataset.
    """

    # Get list of image files

    image_dir = os.path.join(processed_dir, "images")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

    random_seed = 42

    # Split Data
    train_files, temp_files = train_test_split(
        image_files, test_size=(1 - train_split), random_state=random_seed
    )

    val_size = val_split / (val_split + test_split)

    val_files, test_files = train_test_split(
        temp_files, test_size=(1 - val_size), random_state=random_seed
    )

    return train_files, val_files, test_files


def create_data_loaders(
    processed_dir: str,
    num_classes: int,
    batch_size: int,
    train_files: List[str],
    val_files: List[str],
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation

    Parameters
    ----------
    processed_dir: str
        Where the processed dataset should be read from.

    num_classes: int
        Number of classes present in the mask files.

    batch_size: int
        Number of training samples in each batch of training.

    train_files: List[str]
        list of training files paths.

    val_files: List[str]
        list of validation files paths.

    Returns
    -------
    train_data_loader: DataLoader
        Data loader used for training.

    val_data_loader: DataLoader
        Dataloader used for validation.
    """

    image_dir = os.path.join(processed_dir, "images")
    mask_dir = os.path.join(processed_dir, "masks")

    train_dataset = LandCoverDataset(
        image_dir,
        mask_dir,
        train_files,
        transform=get_train_transforms(),
        num_classes=num_classes,
    )

    val_dataset = LandCoverDataset(
        image_dir,
        mask_dir,
        val_files,
        transform=get_val_transforms(),
        num_classes=num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    metrics: SegmentationMetrics,
) -> Tuple[float, Dict[str, Any]]:
    """
    Train for one epoch. Perform a full training epoch, including forward and backward passes,
    optimization steps, and metric updates.

    Parameters
    ----------
    model: nn.Module
        The neural network model to train.

    train_loader: DataLoader
        DataLoader for the training dataset.

    criterion: nn.Module
        Loss function to compute training loss.

    optimizer: optim.Optimizer
        Optimizer for updating model parameters.

    device: torch.device
        Device (CPU/GPU) to perform computations on.

    metrics: SegmentationMetrics
        Metrics tracker for segmentation performance.

    Returns
    -------
    avg_loss: float
        Average loss over the epoch.

    metrics_result: Dict[str, Any]
        Dictionary of computed metrics for the epoch.
    """

    model.train()
    total_loss = 0.0
    metrics.reset()

    pbar = tqdm(train_loader, desc="Training")

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        metrics.update(outputs, masks)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    metrics_result = metrics.get_metrics()

    return avg_loss, metrics_result


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: SegmentationMetrics,
) -> Tuple[float, Dict[str, Any]]:
    """
    Validate for one epoch.

    This function performs a full validation epoch without gradients, computing loss
    and metrics. It uses a progress bar for monitoring.

    Parameters
    ----------
    model: nn.Module
        The neural network model to evaluate.

    val_loader: DataLoader
        DataLoader for the validation dataset.

    criterion: nn.Module
        Loss function to compute validation loss.

    device: torch.device
        Device (CPU/GPU) to perform computations on.

    metrics: SegmentationMetrics
        Metrics tracker for segmentation performance.

    Returns
    -------
    avg_loss: float
        Average loss over the epoch.

    metric_results: Dict[str, Any]
        Dictionary of computed metrics for the epoch.
    """

    model.eval()
    total_loss = 0.0
    metrics.reset()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            metrics.update(outputs, masks)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(val_loader)
    metric_results = metrics.get_metrics()

    return avg_loss, metric_results


def main(args: TrainingArgs):
    device = Config.DEVICE
    print(f"Using device: {device}")

    train_files, val_files, test_files = prepare_data_splits(
        args.processed_dir, args.train_split, args.val_split, args.test_split
    )
    train_loader, val_loader = create_data_loaders(
        processed_dir=args.processed_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        train_files=train_files,
        val_files=val_files,
    )

    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")

    # Initialize model
    model = UNet(args.input_channels, args.num_classes).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Class names for LandCover.ai
    class_names = ["Background", "Buildings", "Woodlands", "Water", "Roads"]
    train_metrics = SegmentationMetrics(args.num_classes, class_names)
    val_metrics = SegmentationMetrics(args.num_classes, class_names)

    # Training loop
    best_val_iou = 0.0
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 50)

        # Train
        train_loss, train_results = train_epoch(
            model, train_loader, criterion, optimizer, device, train_metrics
        )

        # Validate
        val_loss, val_results = validate_epoch(
            model, val_loader, criterion, device, val_metrics
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_results["mean_iou"])
        val_ious.append(val_results["mean_iou"])

        # Print results
        print(
            f"Train Loss: {train_loss:.4f}, Train mIoU: {train_results['mean_iou']:.4f}"
        )

        for class_name in class_names:
            print(f"{class_name} iou: {train_results['class_iou'][class_name]}")

        print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_results['mean_iou']:.4f}")

        if val_results["mean_iou"] > best_val_iou:
            best_val_iou = val_results["mean_iou"]
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved! Val mIoU: {best_val_iou:.4f}")


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the training script.

    Returns:
        argparse.ArgumentParser: Configured parser object.
    """
    parser = argparse.ArgumentParser(description="Train the UNet model.")

    parser.add_argument(
        "--processed-dir",
        default=Config.PROCESSED_DIR,
        type=str,
        help="Where the processed dataset should be read from.",
    )
    parser.add_argument(
        "--train-split",
        default=Config.TRAIN_SPLIT,
        type=float,
        help="The training fraction of the dataset (between 1 and 0).",
    )
    parser.add_argument(
        "--val-split",
        default=Config.VAL_SPLIT,
        type=float,
        help="The validation fraction of the dataset (between 1 and 0).",
    )
    parser.add_argument(
        "--test-split",
        default=Config.TEST_SPLIT,
        type=float,
        help="The Testing fraction of the dataset (between 1 and 0).",
    )
    parser.add_argument(
        "--num-classes",
        default=Config.NUM_CLASSES,
        type=int,
        help="Number of classes present in the mask files.",
    )
    parser.add_argument(
        "--batch-size",
        default=Config.BATCH_SIZE,
        type=int,
        help="Number of training samples in each batch of training.",
    )
    parser.add_argument(
        "--input-channels",
        default=Config.INPUT_CHANNELS,
        type=int,
        help="Number of color channels in the training samples.",
    )
    parser.add_argument(
        "--learning-rate",
        default=Config.LEARNING_RATE,
        type=float,
        help="Models Learning Rate.",
    )
    parser.add_argument(
        "--weight-decay",
        default=Config.WEIGHT_DECAY,
        type=float,
        help="The weight_decay parameter in the Adam optimizer.",
    )
    parser.add_argument(
        "--num-epochs",
        default=Config.NUM_EPOCHS,
        type=int,
        help="Number of training epochs",
    )

    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args(namespace=TrainingArgs())
    main(args)
