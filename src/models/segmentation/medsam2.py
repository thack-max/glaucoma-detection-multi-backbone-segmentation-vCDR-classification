"""
MedSAM-2: Medical Segment Anything Model 2 for volumetric OCT segmentation.

Reference: arXiv:2408.00874 (2024), Jun Ma & Bo Wang (U Toronto / Vector Institute)

Architecture:
  - Built on Meta's SAM 2 (video object segmentation model)
  - Treats 3D OCT volume slices as video frames
  - Memory bank stores features from previously segmented slices
  - One-prompt segmentation: annotate a single slice, propagate to all others

Use case in this pipeline:
  - Segment RNFL (retinal nerve fiber layer) from 3D OCT volumes
  - RNFL thickness is the dominant signal for progression prediction
    (Mass Eye & Ear study: AUC dropped from 0.97 to 0.569 without RNFL)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MedSAM2Segmenter(nn.Module):
    """MedSAM-2 wrapper for volumetric OCT segmentation.

    Treats OCT B-scan stacks as video sequences and uses SAM 2's
    memory-based tracking to propagate segmentation across slices.
    """

    def __init__(
        self,
        sam2_checkpoint: str | None = None,
        target_structures: list[str] | None = None,
    ) -> None:
        """
        Args:
            sam2_checkpoint: Path to SAM 2 pretrained weights.
            target_structures: Structures to segment, e.g. ['rnfl', 'gcl', 'ipl'].
        """
        super().__init__()
        self.target_structures = target_structures or ["rnfl"]

        # TODO: Load SAM 2 model, configure memory bank for slice propagation.
        # The memory bank stores feature embeddings from previously segmented
        # slices, enabling consistent tracking across the full volume.

    def segment_volume(
        self,
        volume: torch.Tensor,
        prompt_slice_idx: int = 0,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Segment a full OCT volume from a single prompted slice.

        Args:
            volume: OCT volume [slices, H, W] or [slices, 1, H, W].
            prompt_slice_idx: Index of the slice with the initial annotation.
            prompt_mask: Segmentation mask for the prompt slice [H, W].

        Returns:
            Full volume segmentation [slices, num_classes, H, W].
        """
        raise NotImplementedError(
            "Full implementation requires SAM 2 backbone. "
            "Install segment-anything-2 and provide checkpoint path."
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Alias for segment_volume with default prompt on middle slice."""
        mid = volume.shape[0] // 2
        return self.segment_volume(volume, prompt_slice_idx=mid)
