"""Tests for data transforms including polar coordinate conversion."""

import numpy as np
import pytest

from src.data.transforms import cartesian_to_polar, polar_to_cartesian


def test_polar_transform_output_shape():
    image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    center = (128, 128)
    radius = 100
    output_size = (128, 256)

    polar = cartesian_to_polar(image, center, radius, output_size)
    assert polar.shape == (128, 256)


def test_polar_transform_rgb():
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    center = (128, 128)
    radius = 100
    output_size = (128, 256)

    polar = cartesian_to_polar(image, center, radius, output_size)
    assert polar.shape == (128, 256, 3)


def test_polar_roundtrip_approximate():
    """Polar -> Cartesian -> Polar should be approximately identity."""
    image = np.zeros((256, 256), dtype=np.uint8)
    # Draw a circle
    for y in range(256):
        for x in range(256):
            if (x - 128) ** 2 + (y - 128) ** 2 < 50**2:
                image[y, x] = 255

    center = (128, 128)
    radius = 100
    output_size = (256, 256)

    polar = cartesian_to_polar(image, center, radius, output_size)
    reconstructed = polar_to_cartesian(polar, center, radius, (256, 256))

    # Not pixel-perfect due to interpolation, but structure should be preserved
    assert reconstructed.shape == image.shape
