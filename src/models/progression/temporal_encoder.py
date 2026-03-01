"""
Temporal feature encoder using ConvNeXt-V2 backbone.

Reference: Mass Eye & Ear longitudinal study (medRxiv, Nov 2025)
  - 10,864 patients
  - ConvNeXt-V2 backbone coupled with BiLSTM
  - AUC 0.97 for 2-4 year progression forecasting

The study tested four backbones: ConvNeXt-V2, ViT, MobileNet-V2, EfficientNet-B0.
ConvNeXt-V2 achieved the best overall performance.

Critical finding: RNFL OCT data is essential (removing it dropped AUC to 0.569),
while visual field data is largely redundant (removing VF: AUC 0.975).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None


class TemporalEncoder(nn.Module):
    """Per-visit feature encoder using ConvNeXt-V2.

    Encodes each imaging observation (OCT RNFL scan) into a fixed-length
    feature vector that is then fed into the BiLSTM forecaster.
    """

    def __init__(
        self,
        backbone: str = "convnextv2_base",
        pretrained: bool = True,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()

        if timm is None:
            raise ImportError("timm is required. Install with: pip install timm")

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )
        # Get actual output dim from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy).shape[-1]

        self.projection = nn.Identity() if backbone_dim == embed_dim else nn.Linear(
            backbone_dim, embed_dim
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single visit's imaging data.

        Args:
            x: Image tensor [B, 3, H, W] (OCT RNFL or fundus).

        Returns:
            Feature vector [B, embed_dim].
        """
        features = self.backbone(x)
        return self.projection(features)

    def encode_sequence(self, image_sequence: torch.Tensor) -> torch.Tensor:
        """Encode a temporal sequence of visits.

        Args:
            image_sequence: [B, T, 3, H, W] where T is number of visits.

        Returns:
            Sequence of feature vectors [B, T, embed_dim].
        """
        B, T, C, H, W = image_sequence.shape
        flat = image_sequence.reshape(B * T, C, H, W)
        features = self.forward(flat)
        return features.reshape(B, T, self.embed_dim)
