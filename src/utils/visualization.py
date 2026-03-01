"""Visualization utilities for segmentation overlays and progression plots."""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def overlay_segmentation(
    image: np.ndarray,
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    disc_color: tuple = (0, 255, 0),
    cup_color: tuple = (255, 0, 0),
    alpha: float = 0.3,
) -> np.ndarray:
    """Overlay optic disc/cup segmentation on fundus image.

    Args:
        image: RGB fundus image [H, W, 3], uint8.
        disc_mask: Binary disc mask [H, W].
        cup_mask: Binary cup mask [H, W].
        disc_color: RGB color for disc boundary.
        cup_color: RGB color for cup boundary.
        alpha: Overlay transparency.

    Returns:
        Image with segmentation overlay [H, W, 3].
    """
    overlay = image.copy().astype(np.float32)

    for mask, color in [(disc_mask, disc_color), (cup_mask, cup_color)]:
        if mask is not None:
            colored = np.zeros_like(overlay)
            for i, c in enumerate(color):
                colored[:, :, i] = c
            mask_3d = np.stack([mask] * 3, axis=-1).astype(np.float32)
            overlay = overlay * (1 - alpha * mask_3d) + colored * alpha * mask_3d

    return overlay.clip(0, 255).astype(np.uint8)


def plot_progression_trajectory(
    visit_dates: list,
    rnfl_values: list[float],
    predicted_values: list[float] | None = None,
    md_values: list[float] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot RNFL thickness trajectory with optional forecast.

    Args:
        visit_dates: List of visit dates/indices.
        rnfl_values: Observed RNFL thickness values (microns).
        predicted_values: Forecasted future values.
        md_values: Visual field Mean Deviation values (dB).
        save_path: Path to save the figure.
    """
    if plt is None:
        raise ImportError("matplotlib required for visualization")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(visit_dates, rnfl_values, "b-o", label="Observed RNFL")
    ax1.set_xlabel("Visit")
    ax1.set_ylabel("RNFL Thickness (μm)", color="b")

    if predicted_values is not None:
        future_dates = list(range(len(visit_dates), len(visit_dates) + len(predicted_values)))
        ax1.plot(future_dates, predicted_values, "b--s", alpha=0.6, label="Predicted RNFL")

    if md_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(visit_dates[: len(md_values)], md_values, "r-^", label="VF MD")
        ax2.set_ylabel("Mean Deviation (dB)", color="r")

    ax1.legend(loc="upper left")
    ax1.set_title("Glaucoma Progression Trajectory")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
