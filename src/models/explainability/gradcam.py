"""
Grad-CAM and Attention Rollout explainability for DINOv2 + classifier pipelines.

Reference papers:
  - Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization." ICCV 2017.
  - Abnar & Zuidema, "Quantifying Attention Flow in Transformers." ACL 2020
    (attention rollout for ViTs).
  - Chefer et al., "Transformer Interpretability Beyond Attention Visualization."
    CVPR 2021 (attention × gradient for ViTs).

Why both methods:
  - **Grad-CAM** (DINOv2GradCAM) is classifier-aware: it highlights regions
    the trained classifier head actually relies on. Requires gradients.
  - **Attention Rollout** (AttentionRolloutCAM) is classifier-free: it rolls
    through the transformer's self-attention weights to show which patches
    the model attends to. Useful when no trained classifier is available
    (e.g. zero-shot screening with just the frozen DINOv2 encoder).

Both produce a [B, H, W] heatmap in [0, 1] that can be blended over the
original fundus image via overlay_heatmap_on_image.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Grad-CAM using patch tokens + classifier gradients
# ---------------------------------------------------------------------------


class DINOv2GradCAM:
    """Classifier-aware Grad-CAM for a DINOv2 encoder + classifier head.

    Works by:
      1. Running encoder.extract_patch_features(image) to get [B, N, D].
      2. Mean-pooling to a [B, D] embedding and feeding it to the classifier.
      3. Backpropping the target logit through the patch features.
      4. Weighting each channel by its mean gradient (standard Grad-CAM).
      5. Summing channels, ReLU, normalizing to [0, 1].
      6. Reshaping the flat [N] vector into a 2D grid and upsampling to the
         original image size.

    The CLS-token path taken by the classifier at runtime is preserved for
    inference; Grad-CAM uses the patch-token path purely as a differentiable
    proxy for explainability.
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier

    def __call__(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """Compute a Grad-CAM heatmap.

        Args:
            image: Input image [B, 3, H, W].
            target_class: Class index to explain. If None, uses the positive
                          class of a binary head (class 1) or the argmax for
                          multi-class.

        Returns:
            Heatmap [B, H, W] in [0, 1], one per batch element.
        """
        # Restore each submodule's own mode afterwards, not just the top-level
        # flag: a wrapper built on the fly (e.g. nn.Sequential(equity, head))
        # starts in training mode, and .train() on it would leave the wrapped
        # inference modules with dropout active.
        modes = {
            m: m.training
            for root in (self.encoder, self.classifier)
            for m in root.modules()
        }
        self.encoder.eval()
        self.classifier.eval()

        try:
            # Grad-CAM needs d(target)/d(patch tokens) only, so the encoder
            # runs without autograd and its patch tokens become a fresh leaf.
            # This works for frozen encoders (the shipped screening config
            # sets freeze_encoder: true) and never touches encoder weights.
            with torch.no_grad():
                patch_feats = self.encoder.extract_patch_features(image)
            if patch_feats.ndim != 3:
                raise ValueError(
                    f"Expected patch features of shape [B, N, D], got {patch_feats.shape}"
                )

            with torch.enable_grad():
                patch_feats = patch_feats.detach().requires_grad_(True)
                # Mean-pool as a differentiable proxy for the CLS token
                cls_embed = patch_feats.mean(dim=1)  # [B, D]
                logits = self.classifier(cls_embed)

                if logits.ndim == 1:
                    logits = logits.unsqueeze(-1)
                target = self._select_target(logits, target_class)
                (grads,) = torch.autograd.grad(
                    target.sum(), patch_feats, allow_unused=True,
                )  # [B, N, D]
        finally:
            for module, was_training in modes.items():
                module.training = was_training  # exact restore, non-recursive

        if grads is None:
            raise RuntimeError(
                "Grad-CAM failed: no gradient flowed to the patch features. "
                "Make sure the classifier head uses its input."
            )

        # Grad-CAM: channel-wise importance = mean gradient across patches
        patch_feats = patch_feats.detach()
        weights = grads.mean(dim=1, keepdim=True)  # [B, 1, D]
        cam = (patch_feats * weights).sum(dim=-1)  # [B, N]
        cam = F.relu(cam)

        return self._reshape_to_2d(cam, image.shape[-2:])

    @staticmethod
    def _select_target(logits: torch.Tensor, target_class: int | None) -> torch.Tensor:
        """Pick a scalar per batch element to backprop from."""
        B, C = logits.shape
        if C == 1:
            return logits.squeeze(-1)
        if target_class is None:
            return logits[:, 1] if C == 2 else logits.gather(1, logits.argmax(dim=1, keepdim=True)).squeeze(-1)
        return logits[:, target_class]

    @staticmethod
    def _reshape_to_2d(
        cam: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Reshape [B, N] to [B, H, W] and normalize to [0, 1]."""
        B, N = cam.shape
        side = int(round(math.sqrt(N)))
        # Trim or pad to the nearest square; ViTs with register tokens may
        # produce a non-square count.
        if side * side != N:
            N_trim = side * side
            cam = cam[:, :N_trim]
            N = N_trim
        cam_2d = cam.view(B, side, side).unsqueeze(1)  # [B, 1, side, side]
        cam_2d = F.interpolate(cam_2d, size=target_size, mode="bilinear", align_corners=False)
        cam_2d = cam_2d.squeeze(1)  # [B, H, W]

        # Per-sample min-max normalization
        flat = cam_2d.view(B, -1)
        mins = flat.min(dim=1, keepdim=True).values
        maxs = flat.max(dim=1, keepdim=True).values
        denom = (maxs - mins).clamp(min=1e-8)
        cam_2d = (flat - mins) / denom
        return cam_2d.view(B, *target_size)


# ---------------------------------------------------------------------------
# Attention rollout (classifier-free)
# ---------------------------------------------------------------------------


class AttentionRolloutCAM:
    """Classifier-free attention visualization for a ViT encoder.

    Rolls the per-layer attention matrices into a single CLS-token attention
    map (Abnar & Zuidema, 2020). When the encoder exposes
    `get_last_selfattention`, we fall back to that single-layer map.

    This is faster than Grad-CAM and requires no classifier, which is ideal
    for the zero-checkpoint "smoke test" in run_screening.py.
    """

    def __init__(self, encoder: nn.Module, head_fusion: str = "mean") -> None:
        """
        Args:
            encoder: The DINOv2-style encoder.
            head_fusion: How to combine per-head attention. One of
                'mean' | 'max' | 'min'.
        """
        self.encoder = encoder
        self.head_fusion = head_fusion

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Compute an attention-based heatmap.

        Args:
            image: [B, 3, H, W].

        Returns:
            Heatmap [B, H, W] in [0, 1].
        """
        self.encoder.eval()
        attn = self._get_attention(image)  # [B, heads, tokens, tokens]
        if attn is None:
            # No attention available — fall back to a blank map
            B, _, H, W = image.shape
            return torch.zeros(B, H, W, device=image.device)

        fused = self._fuse_heads(attn)  # [B, tokens, tokens]

        # CLS token is conventionally index 0; drop it for the map
        cls_attn = fused[:, 0, 1:]  # [B, patches]

        B, N = cls_attn.shape
        side = int(round(math.sqrt(N)))
        if side * side != N:
            # Some variants include register tokens after CLS — trim
            cls_attn = cls_attn[:, : side * side]

        heatmap = cls_attn.view(B, 1, side, side)
        heatmap = F.interpolate(
            heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False,
        ).squeeze(1)

        flat = heatmap.view(B, -1)
        mins = flat.min(dim=1, keepdim=True).values
        maxs = flat.max(dim=1, keepdim=True).values
        heatmap = (flat - mins) / (maxs - mins).clamp(min=1e-8)
        return heatmap.view(B, *image.shape[-2:])

    def _get_attention(self, image: torch.Tensor) -> torch.Tensor | None:
        """Best-effort extraction of the last-layer attention map."""
        if hasattr(self.encoder, "get_attention_maps"):
            try:
                attn = self.encoder.get_attention_maps(image)
                if attn.ndim == 4 and attn.shape[-1] > 1:
                    return attn
            except Exception:
                pass
        inner = getattr(self.encoder, "encoder", None)
        if inner is not None and hasattr(inner, "get_last_selfattention"):
            try:
                return inner.get_last_selfattention(image)
            except Exception:
                return None
        return None

    def _fuse_heads(self, attn: torch.Tensor) -> torch.Tensor:
        if attn.ndim != 4:
            return attn
        if self.head_fusion == "max":
            return attn.max(dim=1).values
        if self.head_fusion == "min":
            return attn.min(dim=1).values
        return attn.mean(dim=1)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


_DEFAULT_JET = np.array(
    [
        [0, 0, 128],
        [0, 0, 255],
        [0, 128, 255],
        [0, 255, 255],
        [128, 255, 128],
        [255, 255, 0],
        [255, 128, 0],
        [255, 0, 0],
        [128, 0, 0],
    ],
    dtype=np.float32,
)


def colorize_heatmap(heatmap: np.ndarray | torch.Tensor) -> np.ndarray:
    """Apply a jet-like colormap to a [H, W] heatmap in [0, 1].

    Returns an [H, W, 3] uint8 RGB image. Uses a bundled colormap so that
    matplotlib is not required at inference time (which matters for rural
    Tier 1 deployments).
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.clip(heatmap, 0.0, 1.0)

    n = _DEFAULT_JET.shape[0]
    idx_f = heatmap * (n - 1)
    idx_lo = np.floor(idx_f).astype(np.int32)
    idx_hi = np.clip(idx_lo + 1, 0, n - 1)
    frac = (idx_f - idx_lo)[..., None]

    c_lo = _DEFAULT_JET[idx_lo]
    c_hi = _DEFAULT_JET[idx_hi]
    out = c_lo * (1 - frac) + c_hi * frac
    return out.clip(0, 255).astype(np.uint8)


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray | torch.Tensor,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a heatmap onto an RGB image.

    Args:
        image: [H, W, 3] uint8 original image.
        heatmap: [H, W] float heatmap in [0, 1] or [H', W'] that will be
                 resized to match the image.
        alpha: Heatmap opacity in [0, 1]. Higher values show the heatmap
               more prominently.

    Returns:
        [H, W, 3] uint8 blended image.
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    if heatmap.shape[:2] != image.shape[:2]:
        # Simple nearest-neighbor resize using numpy (no cv2 dependency).
        h, w = image.shape[:2]
        ys = (np.linspace(0, heatmap.shape[0] - 1, h)).astype(np.int32)
        xs = (np.linspace(0, heatmap.shape[1] - 1, w)).astype(np.int32)
        heatmap = heatmap[ys[:, None], xs[None, :]]

    colored = colorize_heatmap(heatmap).astype(np.float32)
    base = image.astype(np.float32)
    blended = (1 - alpha) * base + alpha * colored
    return np.clip(blended, 0, 255).astype(np.uint8)
