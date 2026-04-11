"""
Tests for ``LlamaVisionReporter`` and its integration with the
specialist pipeline's VLM gate.

The Llama 3.2 Vision weights (~11B params) are not shipped in CI, so
these tests exercise the *unavailable* path exclusively. They lock in
the contract that:

  - ``_ensure_loaded`` raises ``VLMNotConfiguredError`` (a RuntimeError
    subclass) when transformers is missing or weights are inaccessible.
  - ``is_available()`` returns False in that case and caches the result.
  - ``SpecialistPipeline`` with ``enable_vlm=True`` and an unconfigured
    reporter degrades cleanly: it emits a single RuntimeWarning, leaves
    ``clinical_report=None`` on the output, and does NOT crash mid-run.
  - ``enable_vlm=True`` with ``report_generator=None`` warns at
    construction instead of silently disabling the feature.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from src.models.vlm.llama_vision_reporter import (
    LlamaVisionReporter,
    VLMNotConfiguredError,
)
from src.pipeline.specialist import SpecialistPipeline


class TestVLMNotConfiguredError:
    def test_is_runtime_error_subclass(self) -> None:
        """Preserves back-compat for callers that used to catch
        RuntimeError from the old ``_ensure_loaded``."""
        assert issubclass(VLMNotConfiguredError, RuntimeError)


class TestLlamaVisionReporterUnavailable:
    def test_ensure_loaded_raises_when_transformers_missing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Simulate a machine without ``transformers`` installed."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("transformers"):
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        reporter = LlamaVisionReporter()
        with pytest.raises(VLMNotConfiguredError, match="transformers"):
            reporter._ensure_loaded()

    def test_is_available_returns_false_and_caches(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """is_available must not re-probe HuggingFace on every call."""
        import builtins

        real_import = builtins.__import__
        call_count = {"transformers": 0}

        def counting_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("transformers"):
                call_count["transformers"] += 1
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", counting_import)

        reporter = LlamaVisionReporter()
        assert reporter.is_available() is False
        assert call_count["transformers"] == 1
        # Second call must hit the cached error, not re-probe.
        assert reporter.is_available() is False
        assert call_count["transformers"] == 1

    def test_cached_error_reraises_on_second_ensure_loaded(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once a load fails, subsequent ``_ensure_loaded`` calls must
        re-raise ``VLMNotConfiguredError`` without retrying."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("transformers"):
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        reporter = LlamaVisionReporter()
        with pytest.raises(VLMNotConfiguredError):
            reporter._ensure_loaded()
        with pytest.raises(VLMNotConfiguredError, match="previously failed"):
            reporter._ensure_loaded()


class _StubUnavailableReporter:
    """Mimics LlamaVisionReporter's ``generate_report`` in the
    unavailable state — raises ``VLMNotConfiguredError`` on first use."""

    def generate_report(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise VLMNotConfiguredError("stub reporter is never configured")


class TestSpecialistVLMIntegration:
    def test_unconfigured_vlm_warns_once_and_returns_no_report(self) -> None:
        """The pipeline must not crash when VLM is enabled but the
        reporter is unavailable. Instead it should emit a single
        RuntimeWarning and continue."""
        pipeline = SpecialistPipeline(
            report_generator=_StubUnavailableReporter(),
            enable_vlm=True,
            device="cpu",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = pipeline.run(
                fundus_image=torch.zeros(1, 3, 32, 32),
            )
        # No report on the output, and the experimental flag is still
        # set so downstream display code treats it as gated.
        assert out.clinical_report is None
        assert out.vlm_report_is_experimental is True
        # Exactly one VLM warning should have been emitted.
        vlm_warnings = [
            w for w in caught if "VLM report generation disabled" in str(w.message)
        ]
        assert len(vlm_warnings) == 1

        # Second run should NOT re-warn (the warning is one-shot).
        with warnings.catch_warnings(record=True) as caught_again:
            warnings.simplefilter("always")
            pipeline.run(fundus_image=torch.zeros(1, 3, 32, 32))
        assert not [
            w for w in caught_again
            if "VLM report generation disabled" in str(w.message)
        ]

    def test_enable_vlm_without_reporter_warns_at_construction(self) -> None:
        """Catches the silent-disable footgun where a caller passes
        ``enable_vlm=True`` but forgets to also wire in a
        ``report_generator``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SpecialistPipeline(enable_vlm=True, device="cpu")
        assert any(
            "no report_generator" in str(w.message) for w in caught
        )

    def test_disabled_vlm_is_silent(self) -> None:
        """The default path (enable_vlm=False) must not emit any
        VLM-related warnings."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = SpecialistPipeline(
                report_generator=_StubUnavailableReporter(),
                enable_vlm=False,
                device="cpu",
            )
            pipeline.run(fundus_image=torch.zeros(1, 3, 32, 32))
        assert not [w for w in caught if "VLM" in str(w.message)]
