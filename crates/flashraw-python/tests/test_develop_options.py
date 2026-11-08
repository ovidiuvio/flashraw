"""Demosaic and highlight options of the RGB outputs."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import flashraw


def test_edge_aware_and_highlights_options(mlv_path: pathlib.Path) -> None:
    with flashraw.open(mlv_path) as raw:
        base = raw.decode("rgba8")
        # Accepted on any file: patterns without an edge-aware filter set fall back to bilinear.
        edge = raw.decode("rgba8", demosaic="edge-aware")
        assert edge.array.shape == base.array.shape
        neutral = raw.decode("rgba8", highlights="neutral")
        assert neutral.array.shape == base.array.shape
        assert raw.decode("rgba8", highlights="blend").array.shape == base.array.shape
        assert raw.decode("rgba8", white_balance="auto").array.shape == base.array.shape
        assert np.array_equal(raw.decode("rgba8", white_balance="as-shot").array, base.array)
        with pytest.raises(flashraw.InvalidArgumentError):
            raw.decode("rgba8", white_balance="sunny")
        assert np.array_equal(raw.decode("rgba8", highlights="clip").array, base.array)
        with pytest.raises(flashraw.InvalidArgumentError):
            raw.decode("rgba8", demosaic="best")
        with pytest.raises(flashraw.InvalidArgumentError):
            raw.decode("rgba8", highlights="recover")
