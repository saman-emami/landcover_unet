import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import zipfile
import requests
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any, Dict

from config import Config
from models.unet import UNet
from models.losses import CombinedLoss
from utils.dataset import LandCoverDataset
from utils.metrics import SegmentationMetrics
from utils.transforms import get_train_transforms, get_val_transforms


def download_and_extract_dataset() -> None:
    """Download and extract LandCover.ai dataset"""

    os.makedirs(Config.DATA_DIR, exist_ok=True)
    zip_path = os.path.join(Config.DATA_DIR, "landcover.ai.v1.zip")

    # Download
    with requests.get(Config.DATASET_URL, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading..."
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 14):  # 16 KB
                f.write(chunk)
                bar.update(len(chunk))

    # Extract
    with zipfile.ZipFile(zip_path) as zf, tqdm(
        total=len(zf.infolist()), desc="Extracting..."
    ) as bar:
        for member in zf.infolist():
            zf.extract(member, Config.RAW_DIR)
            bar.update(1)


def prepare_data_splits() -> Tuple[List[str], List[str], List[str]]:
    """Prepare train/val/test splits"""

    # Get list of image files

    image_dir = os.path.join(Config.RAW_DIR, "images")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

    random_seed = 40

    # Split Data
    train_files, temp_files = train_test_split(
        image_files, test_size=(1 - Config.TRAIN_SPLIT), random_state=random_seed
    )

    val_size = Config.VAL_SPLIT / (Config.VAL_SPLIT + Config.TEST_SPLIT)
    val_files, test_files = train_test_split(
        temp_files, test_size=(1 - val_size), random_state=random_seed
    )

    return train_files, val_files, test_files


def create_data_loaders(
    train_files: List[str],
    val_files: List[str],
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation"""

    image_dir = os.path.join(Config.RAW_DIR, "images")
    mask_dir = os.path.join(Config.RAW_DIR, "masks")

    train_dataset = LandCoverDataset(
        image_dir,
        mask_dir,
        train_files,
        transform=get_train_transforms(Config.IMG_SIZE),
        num_classes=Config.NUM_CLASSES,
    )

    val_dataset = LandCoverDataset(
        image_dir,
        mask_dir,
        val_files,
        transform=get_val_transforms(Config.IMG_SIZE),
        num_classes=Config.NUM_CLASSES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
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
    """Train for one epoch"""

    model.train()
    total_loss = 0.0
    metrics.reset()

    pbar = tqdm(train_loader, desc="Training...")

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
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    metrics.reset()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating...")
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


def main():
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Download and prepare data
    download_and_extract_dataset()
    train_files, val_files, test_files = prepare_data_splits()
    train_loader, val_loader = create_data_loaders(train_files, val_files)

    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")

    # Initialize model
    model = UNet(Config.INPUT_CHANNELS, Config.NUM_CLASSES).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Class names for LandCover.ai
    class_names = ["Background", "Buildings", "Woodlands", "Water", "Roads"]
    train_metrics = SegmentationMetrics(Config.NUM_CLASSES, class_names)
    val_metrics = SegmentationMetrics(Config.NUM_CLASSES, class_names)

    # Training loop
    best_val_iou = 0.0
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
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
        print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_results['mean_iou']:.4f}")

        if val_results["mean_iou"] > best_val_iou:
            best_val_iou = val_results["mean_iou"]
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved! Val mIoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
