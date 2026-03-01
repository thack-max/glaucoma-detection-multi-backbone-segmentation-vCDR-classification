"""
RETFound: Retinal Foundation Model feature encoder.

Reference: Zhou et al., "A foundation model for generalizable disease detection
from retinal images." Nature 622, 156-163 (2023).
GitHub: https://github.com/rmaphoh/RETFound_MAE

Architecture:
  - ViT-Large backbone
  - Pre-trained via Masked Autoencoder (MAE) on ~1.6M retinal images:
    - 904K color fundus photographs (UK Biobank + EyePACS)
    - 736K OCT images (Moorfields Eye Hospital)
  - Two separate models: RETFound-CFP (fundus) and RETFound-OCT

Use in this pipeline:
  - Phase 3: Extract dense feature embeddings from fundus/OCT images
  - These embeddings capture sub-clinical structural anomalies that explicit
    geometric measurements (like vCDR) cannot represent
  - Embeddings are fed into Phase 4 (BiLSTM) for progression forecasting

Known limitations:
  - RETFound-CFP and RETFound-OCT are separate; no unified multimodal model
  - Pre-training data biased toward UK Biobank demographics
  - Does not fully solve cross-site/cross-device domain shift
  - ViT-Large resolution may lose fine-grained details
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None


class RETFoundEncoder(nn.Module):
    """Feature extractor using RETFound (ViT-Large, MAE pre-trained on retinal images).

    Outputs dense embeddings per image that encode retinal structural features
    far richer than a scalar vCDR value.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        modality: str = "cfp",  # 'cfp' for fundus, 'oct' for OCT
        embed_dim: int = 1024,
        freeze_encoder: bool = False,
    ) -> None:
        """
        Args:
            checkpoint_path: Path to RETFound .pth weights.
                Download from: https://github.com/rmaphoh/RETFound_MAE
            modality: Which RETFound variant ('cfp' or 'oct').
            embed_dim: Output embedding dimension (1024 for ViT-Large).
            freeze_encoder: Whether to freeze all encoder weights.
        """
        super().__init__()
        self.modality = modality
        self.embed_dim = embed_dim

        if timm is None:
            raise ImportError("timm is required for RETFound. Install with: pip install timm")

        # Initialize ViT-Large architecture (same as RETFound)
        self.encoder = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,  # Remove classification head, keep embeddings
        )

        if checkpoint_path:
            self._load_retfound_weights(checkpoint_path)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _load_retfound_weights(self, path: str) -> None:
        """Load RETFound MAE pre-trained weights into the ViT backbone."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)

        # RETFound checkpoints may have 'encoder.' prefix or MAE decoder keys
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state[k.removeprefix("encoder.")] = v
            elif not k.startswith("decoder"):
                encoder_state[k] = v

        msg = self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"RETFound-{self.modality.upper()} loaded. Missing: {len(msg.missing_keys)}, "
              f"Unexpected: {len(msg.unexpected_keys)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract dense embeddings from retinal images.

        Args:
            x: Image tensor [B, 3, 224, 224] (fundus or OCT slice).

        Returns:
            Feature embeddings [B, embed_dim].
        """
        return self.encoder(x)

    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-patch token features (before global pooling).

        Useful for spatial attention maps and VLM input.

        Args:
            x: Image tensor [B, 3, 224, 224].

        Returns:
            Patch tokens [B, num_patches, embed_dim].
        """
        return self.encoder.forward_features(x)
