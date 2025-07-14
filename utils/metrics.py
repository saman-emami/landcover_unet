import torch
import numpy as np
from configs.config import Config
from typing import Optional, Dict, Union, List


def calculate_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    smooth: float = Config.LOSS_SMOOTH,
):
    """
    Compute **Intersection-over-Union (IoU)** for every class.

    Parameters
    ----------
    predictions: torch.Tensor
        Raw model outputs, shape ``[batch_size, H, W]``.

    targets: torch.Tensor
        Ground truth labels with shape ``[batch_size, H, W]``.

    num_classes: int
        Total number of classes including background.

    ignore_index: Optional[int]
        Class ID to ignore (e.g. background). If ``None`` no class is skipped.

    Returns
    -------
    iou_scores: list[float]
        IoU for each class *except* the ignored one.
    """

    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_mask = predictions == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().float().item()
        union = (pred_mask | target_mask).sum().float().item()

        iou = (intersection + smooth) / (union + smooth)

        ious.append(iou)

    return ious


def calculate_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    smooth: float = Config.LOSS_SMOOTH,
):
    """
    Compute Dice coefficient for each class.

    Parameters
    ----------
    predictions: torch.Tensor
        Raw model outputs (logits), shape ``[batch_size, H, W]``.

    targets: torch.Tensor
        Ground truth labels with shape ``[batch_size, H, W]``.

    num_classes: int
        Total number of classes including background.

    ignore_index: Optional[int]
        Class ID to ignore (e.g. background). If ``None`` no class is skipped.

    Returns
    -------
    dice_scores: list[float]
        Dice coefficient for each class *except* the ignored one.
    """
    dice_scores = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_mask = predictions == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().float().item()
        total = (pred_mask.sum() + target_mask.sum()).float().item()

        if total == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection + smooth) / (total + smooth)

        dice_scores.append(dice)

    return dice_scores


class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        This class accumulates IoU and Dice scores over multiple batches and
        computes averages.

        Parameters
        ----------
        num_classes: int
            Total number of classes including background.

        class_names: Optional[List[str]]
            List of class names for reporting. defaults to "Class_0", etc.
        """

        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """
        Clear all accumulated statistics.

        This method resets the internal counters for IoU, Dice, and batch count,
        """
        self.total_iou = [0.0] * self.num_classes
        self.total_dice = [0.0] * self.num_classes
        self.batches = 0

    def update(
        self,
        predictions: torch.Tensor,  # logits [B, C, H, W]
        targets: torch.Tensor,  # one-hot [B, C, H, W]
    ):
        """
        This method computes per-batch IoU and Dice scores and accumulates them.
        It handles conversion from logits and one-hot targets to class indices.

        Parameters
        ----------
        predictions: torch.Tensor
            Model logits with shape [batch_size, num_classes, H, W].

        targets: torch.Tensor
            Ground truth with shape [batch_size, num_classes, H, W] (one-hot).
        """

        predictions = torch.argmax(predictions, dim=1)

        if targets.dim() == 4:  # one-hot
            targets = torch.argmax(targets, dim=1)
        else:
            targets = targets

        iou_scores = calculate_iou(predictions, targets, self.num_classes)
        dice_scores = calculate_dice(predictions, targets, self.num_classes)

        for i in range(self.num_classes):
            self.total_iou[i] += iou_scores[i]
            self.total_dice[i] += dice_scores[i]

        self.batches += 1

    def get_metrics(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Compute mean IoU & Dice and per-class scores.

        This method averages the accumulated scores over all batches and returns
        a dictionary with mean and per-class metrics.

        Returns
        -------
        metrics: Dict[str, Union[float, Dict[str, float]]]
            Dictionary containing 'mean_iou', 'mean_dice', 'class_iou', and 'class_dice'.

        Raises
        ------
        RuntimeError:
            Raises an error if no batches have been processed.
        """
        if self.batches == 0:
            raise RuntimeError("No batches processed. call update() first.")

        avg_iou = [total / self.batches for total in self.total_iou]
        avg_dice = [total / self.batches for total in self.total_dice]

        mean_iou = float(np.mean(avg_iou))
        mean_dice = float(np.mean(avg_dice))

        return {
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "class_iou": dict(zip(self.class_names, avg_iou)),
            "class_dice": dict(zip(self.class_names, avg_dice)),
        }
