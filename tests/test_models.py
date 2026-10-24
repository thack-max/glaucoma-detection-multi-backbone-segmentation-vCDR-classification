"""Tests for the GlaucoMamba model components.

Tests verify model construction, forward pass shapes, and basic functionality
WITHOUT requiring pretrained weights or GPU.
"""

import torch
import pytest


# --- BoundaryFormer ---


class TestBoundaryFormer:
    def test_construction(self):
        from src.models.segmentation.boundary_former import BoundaryFormer
        model = BoundaryFormer(num_classes=3, base_channels=32, growth_rate=16)
        assert model.num_classes == 3

    def test_forward_shape(self):
        from src.models.segmentation.boundary_former import BoundaryFormer
        model = BoundaryFormer(num_classes=3, base_channels=16, growth_rate=8)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "logits" in out
        assert "boundary" in out
        assert out["logits"].shape == (1, 3, 64, 64)
        assert out["boundary"].shape == (1, 1, 64, 64)

    def test_loss_computation(self):
        from src.models.segmentation.boundary_former import (
            BoundaryFormer,
            BoundaryFocalDiceLoss,
        )
        model = BoundaryFormer(num_classes=3, base_channels=16, growth_rate=8)
        loss_fn = BoundaryFocalDiceLoss()
        x = torch.randn(2, 3, 64, 64)
        targets = torch.randint(0, 3, (2, 64, 64))
        out = model(x)
        loss = loss_fn(out, targets)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0


class TestBoundaryAttention:
    def test_shape_preserved(self):
        from src.models.segmentation.boundary_former import BoundaryAttention
        ba = BoundaryAttention(channels=32)
        x = torch.randn(2, 32, 16, 16)
        out = ba(x)
        assert out.shape == x.shape


class TestDepthwiseSeparableConv:
    def test_output_shape(self):
        from src.models.segmentation.boundary_former import DepthwiseSeparableConv
        conv = DepthwiseSeparableConv(16, 32)
        x = torch.randn(1, 16, 32, 32)
        assert conv(x).shape == (1, 32, 32, 32)


# --- Swin-UNETR OCT ---


class TestSwinUNETROCT:
    def test_construction(self):
        from src.models.segmentation.swin_unetr_oct import SwinUNETROCT
        model = SwinUNETROCT(num_classes=4, embed_dim=24, depths=(1, 1, 1, 1), num_heads=(3, 3, 3, 3))
        assert model.num_classes == 4


# --- DINOv2 Encoder ---


class TestDINOv2Encoder:
    def test_construction_fallback(self):
        """Test construction with fallback (no internet/torch.hub)."""
        from src.models.features.dinov2_encoder import DINOv2Encoder
        encoder = DINOv2Encoder(variant="dinov2_vitb14", freeze_encoder=True)
        assert encoder.embed_dim == 768

    def test_variant_validation(self):
        from src.models.features.dinov2_encoder import DINOv2Encoder
        with pytest.raises(ValueError, match="Unknown variant"):
            DINOv2Encoder(variant="invalid_model")

    def test_variants_dict(self):
        from src.models.features.dinov2_encoder import DINOv2Encoder
        assert "dinov2_vits14" in DINOv2Encoder.VARIANTS
        assert "dinov2_vitb14" in DINOv2Encoder.VARIANTS
        assert DINOv2Encoder.VARIANTS["dinov2_vitb14"]["embed_dim"] == 768

    def test_forward_handles_arbitrary_input_sizes(self):
        """DINOv2's timm-backed VIT strictly checks input size; the encoder
        must transparently resize to what the model was built for so that
        downstream pipelines can pass 64x64 test images, 224x224 inference
        crops, or the full 518x518 paper-default."""
        from src.models.features.dinov2_encoder import DINOv2Encoder

        encoder = DINOv2Encoder(variant="dinov2_vitb14", freeze_encoder=True)
        encoder.eval()
        for size in (32, 64, 224):
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                out = encoder(x)
            assert out.shape == (1, encoder.embed_dim)


# --- Delta-t Mamba ---


class TestDeltaTMamba:
    def test_construction(self):
        from src.models.progression.delta_t_mamba import DeltaTMamba
        model = DeltaTMamba(input_dim=64, d_model=32, d_state=8, num_layers=2)
        assert model.output_dim == 64  # bidirectional: 32 * 2

    def test_forward_uniform(self):
        from src.models.progression.delta_t_mamba import DeltaTMamba
        model = DeltaTMamba(input_dim=32, d_model=16, d_state=4, num_layers=1, expand=1)
        model.eval()
        x = torch.randn(2, 5, 32)  # 2 patients, 5 visits
        with torch.no_grad():
            out = model(x)  # No timestamps = uniform spacing
        assert out.shape == (2, 16)  # d_model after output_proj

    def test_forward_with_timestamps(self):
        from src.models.progression.delta_t_mamba import DeltaTMamba
        model = DeltaTMamba(input_dim=32, d_model=16, d_state=4, num_layers=1, expand=1)
        model.eval()
        x = torch.randn(2, 5, 32)
        timestamps = torch.tensor([
            [0, 30, 90, 180, 365],   # Patient 1: irregular visits
            [0, 60, 120, 180, 240],  # Patient 2: more regular
        ], dtype=torch.float32)
        with torch.no_grad():
            out = model(x, timestamps)
        assert out.shape == (2, 16)

    def test_forward_sequence(self):
        from src.models.progression.delta_t_mamba import DeltaTMamba
        model = DeltaTMamba(input_dim=32, d_model=16, d_state=4, num_layers=1, expand=1)
        model.eval()
        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            out = model.forward_sequence(x)
        assert out.shape == (1, 4, 32)  # [B, T, d_model*2]


# --- VF Graph Net ---


class TestVFGraphNet:
    def test_construction(self):
        from src.models.progression.vf_graph_net import VFGraphNet
        model = VFGraphNet(node_features=3, hidden_dim=32, num_gin_layers=2, output_dim=64)
        assert model.output_dim == 64

    def test_adjacency_edges(self):
        from src.models.progression.vf_graph_net import build_adjacency_edges, NUM_VF_POINTS
        edges = build_adjacency_edges()
        assert edges.shape[0] == 2
        assert edges.shape[1] > 0  # Should have edges
        assert edges.max().item() < NUM_VF_POINTS

    def test_forward(self):
        from src.models.progression.vf_graph_net import VFGraphNet
        model = VFGraphNet(node_features=3, hidden_dim=32, num_gin_layers=2, output_dim=32)
        model.eval()
        vf = torch.randn(2, 54, 3)  # 2 patients, 54 VF points, 3 features each
        with torch.no_grad():
            logits = model(vf)
        assert logits.shape == (2, 2)  # binary classification

    def test_node_importance(self):
        from src.models.progression.vf_graph_net import VFGraphNet
        model = VFGraphNet(node_features=3, hidden_dim=32, num_gin_layers=2, output_dim=32)
        model.eval()
        vf = torch.randn(1, 54, 3)
        with torch.no_grad():
            importance = model.get_node_importance(vf)
        assert importance.shape == (1, 54)


# --- Evidential Forecaster ---


class TestEvidentialForecaster:
    def test_construction(self):
        from src.models.progression.evidential_forecaster import EvidentialForecaster
        model = EvidentialForecaster(temporal_dim=64, hidden_dim=32)
        assert model is not None

    def test_forward(self):
        from src.models.progression.evidential_forecaster import EvidentialForecaster
        model = EvidentialForecaster(temporal_dim=64, hidden_dim=32)
        model.eval()
        x = torch.randn(2, 64)
        with torch.no_grad():
            out = model(x)
        assert "classification" in out
        assert "regression" in out
        cls = out["classification"]
        assert cls["prob"].shape == (2, 2)
        assert cls["uncertainty"].shape == (2,)
        assert (cls["uncertainty"] > 0).all()

    def test_forward_with_clinical(self):
        from src.models.progression.evidential_forecaster import EvidentialForecaster
        model = EvidentialForecaster(temporal_dim=64, clinical_dim=8, vf_dim=32, hidden_dim=32)
        model.eval()
        temporal = torch.randn(2, 64)
        clinical = torch.randn(2, 8)
        vf = torch.randn(2, 32)
        with torch.no_grad():
            out = model(temporal, clinical, vf)
        assert out["classification"]["prob"].shape == (2, 2)

    def test_missing_optional_inputs_are_zero_filled(self):
        """A configured clinical/VF slot without data must not change the input width."""
        from src.models.progression.evidential_forecaster import EvidentialForecaster
        model = EvidentialForecaster(temporal_dim=16, clinical_dim=3, vf_dim=8, hidden_dim=16)
        model.eval()
        temporal = torch.randn(2, 16)
        clinical = torch.randn(2, 3)
        with torch.no_grad():
            out = model(temporal)
            explicit = model(temporal, torch.zeros(2, 3), torch.zeros(2, 8))
            clinical_only = model(temporal, clinical_features=clinical)
            clinical_explicit = model(temporal, clinical, torch.zeros(2, 8))
        torch.testing.assert_close(out["classification"]["prob"], explicit["classification"]["prob"])
        torch.testing.assert_close(
            clinical_only["classification"]["prob"], clinical_explicit["classification"]["prob"],
        )

    def test_rejects_mis_sized_or_unconfigured_inputs(self):
        from src.models.progression.evidential_forecaster import EvidentialForecaster
        model = EvidentialForecaster(temporal_dim=16, clinical_dim=3, hidden_dim=16)
        temporal = torch.randn(1, 16)
        with pytest.raises(ValueError, match="clinical"):
            model(temporal, clinical_features=torch.randn(1, 5))
        with pytest.raises(ValueError, match="vf"):
            model(temporal, vf_features=torch.randn(1, 2))  # vf_dim=0
        with pytest.raises(ValueError, match="temporal"):
            model(torch.randn(1, 12))

    def test_evidential_loss(self):
        from src.models.progression.evidential_forecaster import (
            EvidentialForecaster,
            EvidentialLoss,
        )
        model = EvidentialForecaster(temporal_dim=64, hidden_dim=32)
        loss_fn = EvidentialLoss()
        x = torch.randn(4, 64)
        out = model(x)
        cls_targets = torch.randint(0, 2, (4,))
        reg_targets = torch.randn(4) * 50 + 80  # RNFL thickness ~80µm
        loss = loss_fn(out, cls_targets, reg_targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


# --- Cross-Modal Fusion ---


class TestCrossModalFusion:
    def test_construction(self):
        from src.models.fusion.cross_modal_attention import CrossModalFusion
        fusion = CrossModalFusion(
            modality_dims={"fundus": 64, "oct": 64},
            fused_dim=32,
        )
        assert fusion.fused_dim == 32

    def test_forward(self):
        from src.models.fusion.cross_modal_attention import CrossModalFusion
        fusion = CrossModalFusion(
            modality_dims={"fundus": 64, "oct": 64, "clinical": 8},
            fused_dim=32,
            num_heads=2,
        )
        fusion.eval()
        features = {
            "fundus": torch.randn(2, 64),
            "oct": torch.randn(2, 64),
            "clinical": torch.randn(2, 8),
        }
        with torch.no_grad():
            out = fusion(features)
        assert out.shape == (2, 32)

    def test_partial_modalities(self):
        """Test fusion with only a subset of modalities (e.g., no VF data)."""
        from src.models.fusion.cross_modal_attention import CrossModalFusion
        fusion = CrossModalFusion(
            modality_dims={"fundus": 64, "oct": 64, "vf": 32},
            fused_dim=32,
            num_heads=2,
        )
        fusion.eval()
        features = {"fundus": torch.randn(2, 64)}  # Only fundus available
        with torch.no_grad():
            out = fusion(features)
        assert out.shape == (2, 32)

    def test_mc_dropout_uncertainty(self):
        from src.models.fusion.cross_modal_attention import CrossModalFusion
        fusion = CrossModalFusion(
            modality_dims={"fundus": 32, "oct": 32},
            fused_dim=16,
            mc_samples=5,
        )
        features = {"fundus": torch.randn(2, 32), "oct": torch.randn(2, 32)}
        out = fusion(features, return_uncertainty=True)
        assert "mean" in out
        assert "std" in out
        assert out["mean"].shape == (2, 16)
        assert out["std"].shape == (2, 16)
        assert (out["std"] >= 0).all()


# --- Equity Attention ---


class TestEquityModule:
    def test_construction(self):
        from src.models.fairness.equity_attention import EquityModule
        module = EquityModule(feature_dim=64, num_groups=3)
        assert module is not None

    def test_forward_with_groups(self):
        from src.models.fairness.equity_attention import EquityModule
        module = EquityModule(feature_dim=64, num_groups=3)
        features = torch.randn(6, 64)
        groups = torch.tensor([0, 1, 2, 0, 1, 2])
        out = module(features, groups)
        assert out.shape == (6, 64)

    def test_forward_without_groups(self):
        from src.models.fairness.equity_attention import EquityModule
        module = EquityModule(feature_dim=64, num_groups=3)
        features = torch.randn(4, 64)
        out = module(features)
        assert out.shape == (4, 64)

    def test_adversarial_loss(self):
        from src.models.fairness.equity_attention import EquityModule
        module = EquityModule(feature_dim=64, num_groups=3)
        features = torch.randn(6, 64)
        groups = torch.tensor([0, 1, 2, 0, 1, 2])
        loss = module.adversarial_loss(features, groups)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_equity_gap(self):
        from src.models.fairness.equity_attention import EquityModule
        module = EquityModule(feature_dim=64, num_groups=3)
        predictions = torch.tensor([0.8, 0.9, 0.7, 0.6, 0.3, 0.2])
        targets = torch.tensor([1, 1, 1, 1, 0, 0])
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        gap = module.equity_gap(predictions, targets, groups)
        assert "equity_gap" in gap
        assert "min_group_acc" in gap


class TestGradientReversal:
    def test_forward_is_identity(self):
        from src.models.fairness.equity_attention import gradient_reversal
        x = torch.randn(4, 8)
        out = gradient_reversal(x, lambda_=1.0)
        assert torch.allclose(out, x)

    def test_backward_reverses_gradient(self):
        from src.models.fairness.equity_attention import gradient_reversal
        x = torch.randn(4, 8, requires_grad=True)
        out = gradient_reversal(x, lambda_=2.0)
        out.sum().backward()
        # Reversed gradient of sum(x) w.r.t. x is -1 * lambda_ per element
        assert torch.allclose(x.grad, torch.full_like(x, -2.0))

    def test_adversarial_loss_with_grl_updates_features(self):
        """Verify that the GRL path actually pushes gradients back to features."""
        from src.models.fairness.equity_attention import EquityModule

        torch.manual_seed(0)
        module = EquityModule(feature_dim=16, num_groups=3, adversarial_weight=1.0)
        features = torch.randn(6, 16, requires_grad=True)
        groups = torch.tensor([0, 1, 2, 0, 1, 2])

        loss = module.adversarial_loss(features, groups, reverse_gradient=True)
        loss.backward()

        assert features.grad is not None
        # GRL should leave a non-zero gradient on the input features.
        assert features.grad.abs().sum() > 0

    def test_adversarial_loss_without_grl_detaches(self):
        from src.models.fairness.equity_attention import EquityModule

        module = EquityModule(feature_dim=16, num_groups=3)
        features = torch.randn(6, 16, requires_grad=True)
        groups = torch.tensor([0, 1, 2, 0, 1, 2])

        loss = module.adversarial_loss(features, groups, reverse_gradient=False)
        loss.backward()

        # With detach, no gradient should reach the upstream features.
        assert features.grad is None or features.grad.abs().sum() == 0
