"""
DINOv2 feature encoder for retinal image analysis.

Reference: Oquab et al., "DINOv2: Learning Robust Visual Features without
Supervision." Meta AI, 2023. GitHub: facebookresearch/dinov2

Retinal comparison study (Ophthalmology Science, 2025):
  - DINOv2-Base AUC for glaucoma: 0.958 (p < 0.001 vs RETFound 0.940)
  - DINOv2 outperforms RETFound for ocular disease detection
  - Pretrained on 142M natural images — no domain-specific data needed

Key advantages over RETFound:
  - Statistically significantly better for glaucoma detection
  - Well-maintained by Meta AI with extensive community
  - ViT-Base (86M params) vs ViT-Large (307M) — 3.6x smaller
  - Available in multiple sizes (S/B/L/G) for deployment flexibility
  - No separate CFP/OCT weights needed — single unified model

Limitation vs RETFound:
  - RETFound outperforms DINOv2 for systemic disease prediction from retina
    (cardiovascular risk, diabetes). DINOv2's natural image features capture
    morphology but may miss subtle systemic biomarkers.
"""

from __future__ import annotations

import http.client
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_download_error(exc: BaseException | None) -> bool:
    """True for network / IO failures of torch.hub (offline, proxy, HTTP error).

    torch.hub raises those as OSError (URLError, HTTPError, timeouts, ...) or
    http.client errors; offline with no cached repo it raises a RuntimeError
    chained from the URLError. Anything else is not a download problem.
    """
    while exc is not None:
        if isinstance(exc, (OSError, http.client.HTTPException)):
            return True
        exc = exc.__cause__
    return False


def require_pretrained(encoder: nn.Module, allow_random: bool) -> bool:
    """For trainers: stop unless the encoder's pretrained weights loaded.

    A head fitted on a randomly initialised encoder's features is meaningless
    (and would later load without complaint), so training on one requires an
    explicit opt-in. Returns ``encoder.pretrained_loaded``, to be saved in the
    checkpoint as ``"encoder_pretrained"``.

    Raises:
        SystemExit: pretrained weights did not load and ``allow_random`` is False.
    """
    pretrained = bool(getattr(encoder, "pretrained_loaded", True))
    if not pretrained and not allow_random:
        raise SystemExit(
            f"DINOv2 {getattr(encoder, 'variant', '')}: pretrained weights could not be "
            f"downloaded/loaded via torch.hub (see the RuntimeWarning above), so its "
            f"features are random and a model trained on them is meaningless. Fix "
            f"network/proxy access to GitHub and dl.fbaipublicfiles.com (or pre-populate "
            f"the torch.hub cache), or pass --allow-random-encoder for a smoke run (the "
            f"checkpoint then records encoder_pretrained: false)."
        )
    return pretrained


class DINOv2Encoder(nn.Module):
    """Feature extractor using DINOv2 (self-supervised ViT from Meta).

    Provides rich visual features for both fundus and OCT images.
    AUC 0.958 for glaucoma detection — statistically superior to RETFound.
    """

    # Available model variants
    VARIANTS = {
        "dinov2_vits14": {"embed_dim": 384, "params": "22M"},
        "dinov2_vitb14": {"embed_dim": 768, "params": "86M"},
        "dinov2_vitl14": {"embed_dim": 1024, "params": "307M"},
        "dinov2_vitg14": {"embed_dim": 1536, "params": "1.1B"},
    }

    def __init__(
        self,
        variant: str = "dinov2_vitb14",
        freeze_encoder: bool = True,
        use_registers: bool = False,
    ) -> None:
        """
        Args:
            variant: DINOv2 model variant. 'dinov2_vitb14' recommended for
                     best accuracy/efficiency balance (AUC 0.958 for glaucoma).
            freeze_encoder: Whether to freeze all encoder weights.
                           Recommended True for feature extraction.
            use_registers: Use the register-token variant (torch.hub
                          "<variant>_reg") for cleaner attention maps, useful
                          for GradCAM explainability. Its features differ from
                          the plain variant's: heads must be trained on the
                          same choice (the scripts/train_*.py trainers use the
                          plain variant).
        """
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(self.VARIANTS.keys())}")

        self.variant = variant
        # Callable names facebookresearch/dinov2's hubconf.py exports
        # ("_reg4" is timm's naming, not the hub's).
        model_name = f"{variant}_reg" if use_registers else variant
        self.embed_dim = self.VARIANTS[variant]["embed_dim"]

        # Load from torch.hub (downloads automatically on first use).
        # pretrained_loaded tells callers whether the features mean anything.
        self.pretrained_loaded = False
        try:
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=True,
            )
            self.pretrained_loaded = True
        except Exception as exc:  # noqa: BLE001
            if not _is_download_error(exc):
                # A configuration or code error (e.g. a callable the hub does
                # not export): never disguise it as an offline fallback.
                raise
            # Fallback: the architecture without pretrained weights, so tests
            # and offline smoke runs still work. Its features are random:
            # say so loudly instead of silently producing predictions.
            warnings.warn(
                f"DINOv2Encoder: could not download/load pretrained '{model_name}' "
                f"via torch.hub ({type(exc).__name__}: {exc}); falling back to a "
                f"RANDOMLY INITIALISED {variant}. Its features are meaningless "
                f"(fine for tests and offline smoke runs, not for predictions). "
                f"Check pretrained_loaded before trusting outputs.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.encoder = self._build_fallback(variant)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Detect the input size the underlying timm/ViT expects. DINOv2 is
        # commonly wrapped in a timm VisionTransformer whose patch_embed
        # strictly asserts the input spatial size; callers may pass arbitrary
        # sizes (64x64 unit tests, 224x224 inference, 518x518 paper default),
        # so we resize on the fly in forward() to whatever the encoder was
        # built for.
        self._expected_hw: tuple[int, int] | None = None
        patch_embed = getattr(self.encoder, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "img_size"):
            ps = patch_embed.img_size
            if isinstance(ps, (tuple, list)) and len(ps) == 2:
                self._expected_hw = (int(ps[0]), int(ps[1]))
            elif isinstance(ps, int):
                self._expected_hw = (ps, ps)

    def _build_fallback(self, variant: str) -> nn.Module:
        """Build DINOv2 architecture without pretrained weights.

        Uses timm if available, otherwise a simple ViT stub for testing.
        """
        try:
            import timm
            size_map = {
                "dinov2_vits14": "vit_small_patch14_dinov2",
                "dinov2_vitb14": "vit_base_patch14_dinov2",
                "dinov2_vitl14": "vit_large_patch14_dinov2",
            }
            model_name = size_map.get(variant, "vit_base_patch14_dinov2")
            return timm.create_model(model_name, pretrained=False, num_classes=0)
        except ImportError:
            # Minimal stub for import/test without dependencies
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(3, self.embed_dim),
            )

    def _match_input_size(self, x: torch.Tensor) -> torch.Tensor:
        """Resize inputs to the encoder's expected spatial size.

        The torch.hub DINOv2 weights are wrapped in a timm VisionTransformer
        whose patch_embed asserts the input size, so we can't rely on
        "dynamic_img_size" handling being enabled. Resizing to the expected
        H/W up-front is the portable fix and matches DINOv2's standard
        inference convention (224 or 518 square).
        """
        if self._expected_hw is None:
            return x
        eh, ew = self._expected_hw
        if x.shape[-2:] != (eh, ew):
            x = F.interpolate(x, size=(eh, ew), mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embedding.

        Args:
            x: Image tensor [B, 3, H, W]. Will be resized to the encoder's
               expected size before running the forward pass (DINOv2 was
               pretrained at 224/518 square).

        Returns:
            Feature embedding [B, embed_dim].
        """
        x = self._match_input_size(x)
        if hasattr(self.encoder, "forward_features"):
            # timm-style interface
            features = self.encoder.forward_features(x)
            if isinstance(features, dict):
                return features.get("x_norm_clstoken", features.get("x", features))
            # Return CLS token if shape is [B, N+1, D]
            if features.ndim == 3:
                return features[:, 0]
            return features
        else:
            # torch.hub DINOv2 interface
            return self.encoder(x)

    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-patch token features (before CLS pooling).

        Useful for:
          - GradCAM attention visualization (explainability)
          - Spatial feature maps for segmentation guidance
          - Dense prediction tasks

        Args:
            x: Image tensor [B, 3, H, W].

        Returns:
            Patch tokens [B, num_patches, embed_dim] (excluding CLS token).
        """
        x = self._match_input_size(x)
        if hasattr(self.encoder, "forward_features"):
            features = self.encoder.forward_features(x)
            if isinstance(features, dict):
                patchtokens = features.get("x_norm_patchtokens", None)
                if patchtokens is not None:
                    return patchtokens
            if features.ndim == 3 and features.shape[1] > 1:
                return features[:, 1:]  # Skip CLS token
            return features
        elif hasattr(self.encoder, "get_intermediate_layers"):
            # DINOv2 torch.hub provides this
            output = self.encoder.get_intermediate_layers(x, n=1)[0]
            return output[:, 1:]  # Skip CLS
        else:
            return self.forward(x).unsqueeze(1)

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract self-attention maps for GradCAM-style explainability.

        This provides visual interpretability that vCDR-based methods have
        by default — addressing the black-box criticism of foundation models.

        Args:
            x: Image tensor [B, 3, H, W].

        Returns:
            Attention maps [B, num_heads, num_patches, num_patches].
        """
        x = self._match_input_size(x)
        if hasattr(self.encoder, "get_last_selfattention"):
            return self.encoder.get_last_selfattention(x)

        # Fallback: hook-based extraction
        attn_maps = []

        def hook_fn(module, input, output):
            # Standard attention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) > 1:
                attn_maps.append(output[1])

        # Register hook on last attention layer
        hooks = []
        for module in self.encoder.modules():
            if isinstance(module, nn.MultiheadAttention):
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            self.encoder(x)

        for h in hooks:
            h.remove()

        if attn_maps:
            return attn_maps[-1]
        return torch.zeros(x.shape[0], 1, 1, 1, device=x.device)
