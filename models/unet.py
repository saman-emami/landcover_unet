import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv -> BatchNorm -> ReLU) * 2

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Applies two consecutive Conv-BN-ReLU blocks to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, H, W].
        """
        return self.conv(x)


class DownSample(nn.Module):
    """
    Downsampling block: MaxPool2d -> DoubleConv

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        Applies 2x2 max pooling followed by a double convolution.

        Parameters:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, H/2, W/2].
        """
        return self.pool_conv(x)


class UpSample(nn.Module):
    """
    Upsampling block: TransposedConv -> Concatenate -> DoubleConv

    Parameters:
        in_channels (int): Number of input channels (from previous decoder layer).
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Upsamples x1, pads if necessary, concatenates with x2, then applies double convolution.

        Parameters:
            x1 (torch.Tensor): Upsampled features, shape [batch_size, in_channels, H, W].
            x2 (torch.Tensor): Skip connection features, shape [batch_size, in_channels // 2, H*2, W*2].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, H*2, W*2].
        """

        # x1: upsampled features, x2: skip connection features
        x1 = self.up(x1)

        diff_y = x1.size(dim=2) - x2.size(dim=2)
        diff_x = x2.size(dim=3) - x2.size(dim=3)

        left_padding = diff_x // 2
        right_padding = diff_x - left_padding
        up_padding = diff_y // 2
        down_padding = diff_y - up_padding

        x1 = F.pad(x1, [left_padding, right_padding, up_padding, down_padding])

        x = torch.cat((x1, x2), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture for semantic segmentation.

    Parameters:
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        num_classes (int): Number of output classes for segmentation.
    """

    def __init__(self, in_channels, num_classes=5):
        super().__init__()

        # Encoder path
        self.input_conv = DoubleConv(in_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)

        # Decoder path
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        # Output layer
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the UNet model.

        Parameters:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes, H, W].
        """
        # Encoder
        x1 = self.input_conv(x)  # [B, 64, H, W]
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        x4 = self.down3(x3)  # [B, 512, H/8, W/8]
        x5 = self.down4(x4)  # [B, 1024, H/16, W/16]

        # Decoder with skip connections
        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)  # [B, 256, H/4, W/4]
        x = self.up3(x, x2)  # [B, 128, H/2, W/2]
        x = self.up4(x, x1)  # [B, 64, H, W]

        # Output segmentation map
        output = self.output_conv(x)  # [B, num_classes, H, W]
        return output
