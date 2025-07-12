import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.

    Parameters:
        smooth (float): Small value added for numerical stability to avoid division by zero.
    """

    def __init__(self, smooth: float = Config.LOSS_SMOOTH):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Parameters:
            predictions (torch.Tensor): Raw model outputs (logits), shape ``[batch_size, num_classes, H, W]``.
            targets (torch.Tensor): Ground truth labels with shape ``[batch_size, num_classes, H, W]``.

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice coefficient).
        """
        predictions = F.softmax(predictions, dim=1)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        total = predictions.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)

        return 1 - dice


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss for semantic segmentation.

    Parameters:
        smooth (float): Small value added for numerical stability to avoid division by zero.
    """

    def __init__(self, smooth: float = Config.LOSS_SMOOTH):
        super().__init__()

        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Parameters:
            predictions (torch.Tensor): Raw model outputs (logits), shape ``[batch_size, num_classes, H, W]``.
            targets (torch.Tensor): Ground truth labels with shape ``[batch_size, num_classes, H, W]``.

        Returns:
            torch.Tensor: Scalar loss value (1 - IoU coefficient).
        """
        predictions = F.softmax(predictions, dim=1)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou


class CombinedLoss(nn.Module):
    """
    Combined CrossEntropy and Dice Loss for semantic segmentation.

    Parameters:
        ce_weight (float): Weight for CrossEntropy loss component.
        dice_weight (float): Weight for Dice loss component.
    """

    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(
        self,
        predictions: torch.Tensor,  # logits  [B, C, H, W]
        targets: torch.Tensor,  # one-hot [B, C, H, W]
    ) -> torch.Tensor:
        """
        Parameters:
            predictions (torch.Tensor): Raw model outputs (logits), shape [batch_size, num_classes, H, W].
            targets (torch.Tensor): Ground truth labels with shape [batch_size, num_classes, H, W].

        Returns:
            torch.Tensor: Scalar loss value (weighted sum).
        """

        ce_targets = targets.argmax(dim=1)  # -> [B, H, W] integer labels
        ce = self.cross_entropy_loss(predictions, ce_targets)
        dice = self.dice_loss(predictions, targets)

        return self.ce_weight * ce + self.dice_weight * dice
