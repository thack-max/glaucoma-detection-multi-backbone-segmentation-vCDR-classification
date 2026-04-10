"""
BoundaryFormer: Dense Depthwise Separable U-Net with boundary attention for
optic disc and cup segmentation.

Architecture inspired by:
  - DDSC-Net (BMC Medical Imaging, 2020): Dense depthwise separable convolutions
    for OD/OC segmentation. REFUGE: Dice OD ~96%, OC ~89%.
  - ODCS-NSNP (2023): Boundary-enhanced segmentation achieving Dice OC 91% on REFUGE.

Key design decisions:
  - Depthwise separable convolutions reduce parameters ~9x vs standard convs
    while maintaining or improving segmentation quality.
  - Dense connections within each encoder/decoder block provide gradient flow
    and multi-scale feature reuse (critical for cup boundaries).
  - Boundary attention module applies a learned edge detector that up-weights
    features near disc/cup boundaries — the hardest region to segment.
  - Combined focal + dice loss addresses the severe class imbalance between
    background, disc, and cup pixels.

Target REFUGE metrics (based on published literature):
  - Dice OD: ~96%    |  IoU OD: ~89%
  - Dice OC: ~89%+   |  IoU OC: ~81%+  (improvement over FunduSAM's 86.7%)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise.

    ~9x fewer parameters than standard convolution for 3x3 kernels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class DenseBlock(nn.Module):
    """Dense block with depthwise separable convolutions.

    Each layer receives concatenated outputs from all previous layers,
    providing multi-scale feature reuse within the block.
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int = 32,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_channels + i * growth_rate
            self.layers.append(DepthwiseSeparableConv(layer_in, growth_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class BoundaryAttention(nn.Module):
    """Boundary attention module for enhancing disc/cup edge features.

    Learns to detect boundary regions and applies attention weighting
    to amplify features near object boundaries — where segmentation
    errors are most common (especially for the ambiguous cup margin).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Boundary detector: Laplacian-like learned edge filter
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
        )
        # Attention gate
        self.attention = nn.Sequential(
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_features = self.edge_conv(x)
        boundary_map = self.attention(edge_features)
        refined = self.refine(x)
        return x + refined * boundary_map


class EncoderBlock(nn.Module):
    """Encoder block: DenseBlock + downsampling."""

    def __init__(self, in_channels: int, out_channels: int, growth_rate: int = 32, num_layers: int = 4) -> None:
        super().__init__()
        self.dense = DenseBlock(in_channels, growth_rate, num_layers)
        dense_out = in_channels + num_layers * growth_rate
        self.transition = nn.Sequential(
            nn.Conv2d(dense_out, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dense_out = self.dense(x)
        features = self.transition(dense_out)
        pooled = self.pool(features)
        return pooled, features  # pooled for next stage, features for skip


class DecoderBlock(nn.Module):
    """Decoder block: upsample + skip concat + DenseBlock + boundary attention."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_boundary_att: bool = True) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        merged_channels = in_channels // 2 + skip_channels
        self.dense = DenseBlock(merged_channels, growth_rate=32, num_layers=3)
        dense_out = merged_channels + 3 * 32
        self.transition = nn.Sequential(
            nn.Conv2d(dense_out, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.boundary_att = BoundaryAttention(out_channels) if use_boundary_att else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.dense(x)
        x = self.transition(x)
        x = self.boundary_att(x)
        return x


class BoundaryFormer(nn.Module):
    """Dense Depthwise Separable U-Net with boundary attention.

    A U-Net variant that uses dense depthwise separable convolution blocks
    in both encoder and decoder, with boundary attention modules in the
    decoder to enhance disc/cup edge segmentation.

    Designed to improve on FunduSAM's weak optic cup segmentation (86.7% Dice)
    through boundary-aware feature refinement and efficient dense connectivity.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,  # background, disc, cup
        base_channels: int = 64,
        growth_rate: int = 32,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )

        # Encoder path
        ch = base_channels
        self.enc1 = EncoderBlock(ch, ch * 2, growth_rate)        # -> ch*2
        self.enc2 = EncoderBlock(ch * 2, ch * 4, growth_rate)    # -> ch*4
        self.enc3 = EncoderBlock(ch * 4, ch * 8, growth_rate)    # -> ch*8
        self.enc4 = EncoderBlock(ch * 8, ch * 16, growth_rate)   # -> ch*16

        # Bottleneck
        self.bottleneck = DenseBlock(ch * 16, growth_rate, num_layers=6)
        bottleneck_out = ch * 16 + 6 * growth_rate
        self.bottleneck_transition = nn.Sequential(
            nn.Conv2d(bottleneck_out, ch * 16, 1, bias=False),
            nn.BatchNorm2d(ch * 16),
            nn.GELU(),
        )

        # Decoder path — boundary attention at every level.
        # Skip connections come from the encoder transition outputs, which
        # match the encoder's out_channels at each level.
        self.dec4 = DecoderBlock(ch * 16, ch * 16, ch * 8, use_boundary_att=True)
        self.dec3 = DecoderBlock(ch * 8, ch * 8, ch * 4, use_boundary_att=True)
        self.dec2 = DecoderBlock(ch * 4, ch * 4, ch * 2, use_boundary_att=True)
        self.dec1 = DecoderBlock(ch * 2, ch * 2, ch, use_boundary_att=True)

        # Segmentation head
        self.seg_head = nn.Conv2d(ch, num_classes, 1)

        # Auxiliary boundary head (for deep supervision during training)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch // 2),
            nn.GELU(),
            nn.Conv2d(ch // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Fundus image tensor [B, 3, H, W].

        Returns:
            Dict with:
              - 'logits': Segmentation logits [B, num_classes, H, W]
              - 'boundary': Boundary prediction [B, 1, H, W] (for training)
        """
        # Encoder
        x0 = self.stem(x)
        x1, skip1 = self.enc1(x0)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)

        # Bottleneck
        b = self.bottleneck(x4)
        b = self.bottleneck_transition(b)

        # Decoder
        d4 = self.dec4(b, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        # Heads
        logits = self.seg_head(d1)
        boundary = self.boundary_head(d1)

        # Resize to match input if needed
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
            boundary = F.interpolate(boundary, size=x.shape[2:], mode="bilinear", align_corners=False)

        return {"logits": logits, "boundary": boundary}


class BoundaryFocalDiceLoss(nn.Module):
    """Combined focal + dice + boundary loss for OD/OC segmentation.

    The weighted combination addresses:
      - Focal loss: class imbalance (background >> disc >> cup)
      - Dice loss: overlap-based optimization matching the evaluation metric
      - Boundary loss: penalizes errors near disc/cup edges specifically
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Multi-class focal loss."""
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.focal_alpha * (1 - pt) ** self.focal_gamma * ce).mean()

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-class soft dice loss, averaged over classes."""
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dice_total = 0.0
        for c in range(num_classes):
            p = probs[:, c].flatten(1)
            t = targets_onehot[:, c].flatten(1)
            intersection = (p * t).sum(1)
            dice = (2.0 * intersection + 1e-6) / (p.sum(1) + t.sum(1) + 1e-6)
            dice_total += (1 - dice).mean()

        return dice_total / num_classes

    def boundary_loss(
        self,
        boundary_pred: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Binary boundary loss — supervises the auxiliary boundary head."""
        # Generate boundary ground truth from segmentation labels
        # Boundary = pixels adjacent to class transitions
        kernel = torch.ones(1, 1, 3, 3, device=targets.device)
        targets_float = targets.unsqueeze(1).float()
        dilated = F.conv2d(targets_float, kernel, padding=1)
        eroded = -F.conv2d(-targets_float, kernel, padding=1)
        boundary_gt = (dilated != eroded).float()

        return F.binary_cross_entropy_with_logits(boundary_pred, boundary_gt)

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            model_output: Dict from BoundaryFormer.forward().
            targets: Ground truth segmentation [B, H, W] with class indices.

        Returns:
            Weighted combined loss.
        """
        logits = model_output["logits"]
        boundary_pred = model_output["boundary"]

        loss = (
            self.focal_weight * self.focal_loss(logits, targets)
            + self.dice_weight * self.dice_loss(logits, targets)
            + self.boundary_weight * self.boundary_loss(boundary_pred, targets)
        )
        return loss
